---
name: stop_experiment_v2
overview: 在根目录新建 stop_experiment 文件夹，构建 Stop-Loss Clustering 策略的 GBDT 实验框架。核心目标：训练两个 LightGBM 模型（卖点模型=max_high_ret_20 / 买点模型=min_low_ret_20），评估因子重要性，指导模型设计。数据源仅用 stop_loss_selection 表（204天，14728条），不用 factor_value 表（仅3天数据）。Pipeline 简化为5步。
todos:
  - id: create-structure
    content: 创建 stop_experiment 目录结构及基础模块（stop_config.py, factor_columns.py, derived_features.py）
    status: completed
  - id: build-01-candidate
    content: 实现 01_build_candidate_table.py：读stop_loss_selection + 计算MFE/MAE多窗口标签
    status: completed
    dependencies:
      - create-structure
  - id: build-02-gbdt
    content: 实现 02_train_gbdt_models.py：卖点模型(mfe_20)+买点模型(mae_20)+多窗口对比训练
    status: completed
    dependencies:
      - build-01-candidate
  - id: build-03-importance
    content: 实现 03_factor_importance.py：因子重要性分析+SHAP+可视化
    status: completed
    dependencies:
      - build-02-gbdt
  - id: build-04-05-decision
    content: 实现 04_signal_selector.py + 05_daily_trading_sheet.py：信号精选+每日决策
    status: completed
    dependencies:
      - build-03-importance
  - id: build-runner-notes
    content: 实现 run_daily.py + EXPERIMENT_NOTES.md + 全流程自测验证
    status: completed
    dependencies:
      - build-04-05-decision
---

## 产品概述

在根目录新建 `stop_experiment` 文件夹，参考 `dsa_experiment` 架构，设计 Stop-Loss Clustering（止损聚类）策略的 GBDT 实验框架。

**核心目标**：SLC信号触发后进入最长20天观察期，观察期内每天用两个模型预测买点和卖点时机，评估因子重要性，指导模型精简设计。

## 核心特性

- **信号 ≠ 买入**：SLC信号触发只代表进入观察期（最长20天），不是直接买入信号
- **每日预测**：观察期内每个交易日都运行模型，评估当天是否为好的买点/卖点
- **两个模型**：卖点模型（找20天内最高点）+ 买点模型（找20天内最低点）
- **回归+分类双测**：回归预测 MFE/MAE 连续值，分类预测是否超过阈值（涨幅>7% / 跌幅>3%）
- **因子重要性分析**：gain/split 重要性 + SHAP 可解释性 + 双模型对比，输出因子排名和可视化
- **5步Pipeline**：候选表构建（展开20天观察期）→ GBDT训练（4个模型）→ 因子重要性分析 → 信号精选 → 每日决策

## 数据现状

| 数据源 | 时间范围 | 数据量 | 可用性 |
| --- | --- | --- | --- |
| `stop_loss_selection` | 2025-07-01 ~ 2026-05-06 | 14728条，2028只股票，204个交易日 | 主力数据（信号源） |
| `factor_value_1d` | 2025-10-28 ~ 2026-05-06 | 125天，4854只股票，85个因子 | **可用**（10-11月仅105只，12月起4854只） |
| `factor_value`(旧分区表) | 2026-04-28 ~ 2026-04-30 | 仅3天 | 不可用（已被新表取代） |
| `stock_k_data` | 2018-05-18 ~ 2026-05-06 | 5018只股票 | 用于计算标签+动态特征+实时因子计算 |


**factor_value_1d 说明**：

- 新表（按月RANGE分区），2025-10-28起有数据
- 2025-10~11月仅覆盖105只股票，2025-12月起覆盖4854只
- 与 stop_loss_selection 重叠区间：2025-10-28 ~ 2026-05-06（8689条信号，59%）
- 2025-07~10-27 的6039条信号（41%）没有 factor_value_1d 数据，需要实时计算

**因子计算函数引用策略**：

- 不直接读 factor_value_1d 表，而是调用核心计算函数实时计算（`compute_dsa` / `compute_bbmacd` / `volume_zscore`）
- 这样所有日期的因子口径统一，不受 factor_value_1d 覆盖限制
- 构建完成后随机抽样验证与 factor_value_1d 一致性

## 模型设计

### 两个模型

| 模型 | 回归目标 | 分类目标 | 实际含义 |
| --- | --- | --- | --- |
| **卖点模型** | `mfe_20` = 未来20天 max(high)/当日close - 1 | `mfe_20 > 7%` → 还有上涨空间 | 找20天内最高点作为卖点 |
| **买点模型** | `mae_20` = 未来20天 min(low)/当日close - 1 | `mae_20 < -3%` → 还有下跌风险 | 找20天内最低点作为买点 |


### 分类阈值解读

- **卖点模型分类**：`mfe_20 > 7%` = 正类 → 未来20天还能涨7%以上，说明还没到最高点，**不卖**
- 负类 = 上涨空间不足7% → 可能接近顶部，**考虑卖出**
- **买点模型分类**：`mae_20 < -3%` = 正类 → 未来20天还可能跌3%以上，**不买**
- 负类 = 下跌风险不足3% → 可能接近底部，**考虑买入**

### 4个模型变体

| 编号 | 模型类型 | 目标变量 | LightGBM objective |
| --- | --- | --- | --- |
| sell_reg | 卖点回归 | mfe_20（连续值） | regression |
| sell_cls | 卖点分类 | mfe_20 > 7%（二分类） | binary |
| buy_reg | 买点回归 | mae_20（连续值） | regression |
| buy_cls | 买点分类 | mae_20 < -3%（二分类） | binary |


## 观察期展开设计

每条 stop_loss_selection → 展开为20天观察期 → 每天一个样本 → 约29万样本

```
信号触发日: selection_date = 2025-08-01
观察期:     2025-08-04(D1), 2025-08-05(D2), ..., 2025-08-28(D20)
                    ↓              ↓                    ↓
               sample_1       sample_2            sample_20

每个sample的标签:
  mfe_20 = max(high[D+1 : D+20]) / close[D] - 1
  mae_20 = min(low[D+1 : D+20]) / close[D] - 1
```

## 特征体系（核心改动：引入因子库 + 剔除不可比因子）

### 设计原则

1. **只保留可跨股票比较的因子**：归一化/百分比/z-score/比率/计数/布尔
2. **剔除绝对值因子**：绝对价格、绝对成交量、绝对波动值等，因不同股票流通值和价格量级不同
3. **引入因子库**：趋势/位置/动量/量能 4大类，调用核心计算函数实时计算
4. **SSOT**：因子计算不重写，引用 `features/` 和 `factor_lib/` 的权威实现

### 一、SLC静态特征（来自stop_loss_selection，观察期内不变）

**保留（可跨股票比较）- 12列**：

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| sell_stop_triggered | bool | 卖止损是否触发 |
| buy_stop_triggered | bool | 买止损是否触发 |
| active_sell_cluster_count | int | 活跃卖cluster数（计数，可比） |
| active_buy_cluster_count | int | 活跃买cluster数（计数，可比） |
| dist_to_nearest_sell_stop_atr | float | 到卖止损的ATR归一化距离（可比） |
| dist_to_nearest_buy_stop_atr | float | 到买止损的ATR归一化距离（可比） |
| stop_cluster_ratio | float | sell/buy活跃量比（比率，可比） |
| change_pct | float | 信号日涨跌幅（百分比，可比） |
| vol_zscore | float | 信号日成交量Z-Score（z-score，可比） |
| daily_bb_width_zscore | float | 信号日布林带宽度Z-Score（z-score，可比） |
| last_event_bars_ago | int | 上次触发距今bar数（计数，可比） |
| last_event_type | cat | 上次触发类型（分类，label encode） |


**剔除（不可跨股票比较）- 9列**：

| 因子名 | 原因 |
| --- | --- |
| ~~sell_trigger_volume~~ | 绝对成交量，不同股票量级差万倍 |
| ~~buy_trigger_volume~~ | 绝对成交量 |
| ~~sum_sells_active~~ | 绝对成交量 |
| ~~sum_buys_active~~ | 绝对成交量 |
| ~~sell_trigger_max_vol_price~~ | 绝对价格 |
| ~~sell_stop_scale~~ | 绝对价格×成交量 |
| ~~nearest_sell_stop_price~~ | 绝对价格 |
| ~~nearest_buy_stop_price~~ | 绝对价格 |
| ~~last_event_volume~~ | 绝对成交量 |


### 二、因子库特征（调用核心计算函数实时计算）

**来源**：`features/dsa_bbmacd_24factors_viewer.py`（compute_dsa + compute_bbmacd）+ `features/volume_zscore_plotly.py`（volume_zscore）

**只选取可跨股票比较的因子，同样剔除绝对值因子**：

#### 趋势类（4列，全部可比）

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| dsa_dir | int | DSA趋势方向（1=上升，-1=下降） |
| prev_pivot_code | int | 前枢轴编码（HH=2,HL=1,LH=-1,LL=-2） |
| trend_align_momo | int | 趋势-动量一致性（1=一致，-1=背离） |
| dsa_dir_age | int | DSA方向持续K线数 |


#### 位置类（6列，全部可比）

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| dsa_pivot_pos_01 | float | DSA枢轴位置（0~1归一化） |
| price_vs_dsa_vwap_pct | float | 价格相对DSA VWAP百分比 |
| ret_to_last_high_pct | float | 距前高回撤百分比 |
| ret_to_last_low_pct | float | 距前低反弹百分比 |
| current_pullback_from_stage_extreme_pct | float | 当前阶段极端回撤百分比 |
| liquidity_range_pos_01 | float | 流动性区间位置（0~1归一化） |


#### 动量类（9列中选5列可比）

**保留**：

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| bbmacd_band_pos_01 | float | BBMACD布林带位置（0~1归一化） |
| bbmacd_bandwidth_zscore | float | BBMACD带宽Z-score |
| bbmacd_sign | int | BBMACD符号（1/0/-1） |
| bbmacd_slope_3 | float | BBMACD 3日斜率 | ← 严格来说是绝对值，但3日差分缩小了量级差异，暂保留观察 |
| bbmacd_state | int | BBMACD状态编码 |


**剔除（绝对值，不可跨股票比较）**：

| 因子名 | 原因 |
| --- | --- |
| ~~bbmacd~~ | 绝对MACD值，600519=-16 vs 000001=0.16 |
| ~~bbmacd_minus_avg~~ | 绝对偏差值 |
| ~~bbmacd_avg~~ | 绝对信号线值 |
| ~~bbmacd_std~~ | 绝对标准差 |


**布尔信号保留**（跨股票可比）：

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| bbmacd_cross_upper | bool | 上穿布林带上轨 |
| bbmacd_cross_lower | bool | 下穿布林带下轨 |
| bbmacd_cross_up_lower | bool | 从下方穿上轨 |
| bbmacd_cross_down_upper | bool | 从上方穿下轨 |


#### 量能类（8列，全部可比）

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| vol_zscore_5 | float | 5日成交量Z-score |
| vol_zscore_10 | float | 10日成交量Z-score |
| vol_zscore_20 | float | 20日成交量Z-score |
| vol_ratio_10 | float | 10日成交量比率 |
| days_since_vol_spike | int | 距上次放量天数 |
| vol_stage_cv | float | 当前阶段量CV |
| vol_prev_stage_cv | float | 前一阶段量CV |
| vol_cv_ratio | float | CV比率 |


#### 风险类（4列，全部可比）

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| atr_pct | float | ATR占价格百分比（归一化） |
| volatility_20d | float | 20日波动率百分比 |
| max_drawdown_60d | float | 60日最大回撤百分比 |
| beta | float | Beta系数 |


#### 节奏/协同类（6列可比）

| 因子名 | 类型 | 说明 |
| --- | --- | --- |
| current_stage_bars | int | 当前阶段K线数 |
| current_stage_amp_pct | float | 当前阶段振幅百分比 |
| current_stage_ret_pct | float | 当前阶段收益百分比 |
| prev_stage_bars | int | 前一阶段K线数 |
| prev_stage_amp_pct | float | 前一阶段振幅百分比 |
| price_vol_coord | int | 价量协同方向 |


**协同/结构类剔除（绝对值）**：

| 因子名 | 原因 |
| --- | --- |
| ~~coord_consistency~~ | 需确认是否归一化，暂剔除 |
| ~~coord_stage_current~~ | 绝对值 |
| ~~coord_stage_prev~~ | 绝对值 |
| ~~coord_stage_ratio~~ | 可能是比率，暂保留观察 |
| ~~support_resistance_zones~~ | 绝对价格 |
| ~~trendline_upper/lower~~ | 绝对价格 |
| ~~liquidity_pools~~ | 绝对价格 |
| ~~upper_break/lower_break~~ | 布尔，但需确认含义 |
| ~~last_confirmed_high/low~~ | 绝对价格 |
| ~~DSA_VWAP~~ | 绝对价格 |
| ~~m_rapida/m_lenta~~ | 绝对移动均线 |
| ~~dsa_atr~~ | 绝对ATR值（用atr_pct替代） |


### 三、动态特征（每个观察日从K线计算）- 10列

| 因子名 | 说明 | 计算方式 |
| --- | --- | --- |
| obs_day | 观察期第几天 | 1~20 |
| ret_to_trigger | 相对信号日涨跌幅 | (close[D] - close[trigger]) / close[trigger] |
| high_to_trigger | 观察期内最高价/信号日收盘价 | max(high[D1:D]) / close[trigger] |
| low_to_trigger | 观察期内最低价/信号日收盘价 | min(low[D1:D]) / close[trigger] |
| intraday_range | 当天日内振幅 | (high[D] - low[D]) / close[D] |
| vol_ratio | 当天成交量/20日均量 | volume[D] / mean(volume[D-20:D]) |
| range_position | 收盘价在区间高低中位置 | (close[D] - min_low) / (max_high - min_low) |
| vol_change | 成交量变化率 | volume[D] / volume[D-1] - 1 |
| dist_to_sell_stop_pct | 当前价到卖止损距离(%) | (close[D] - nearest_sell_stop_price) / close[D] |
| dist_to_buy_stop_pct | 当前价到买止损距离(%) | (close[D] - nearest_buy_stop_price) / close[D] |


**注意**：`dist_to_sell_stop_pct` 和 `dist_to_buy_stop_pct` 使用百分比形式（距离/当前价），替代了旧版的绝对距离。虽然 nearest_sell_stop_price 本身是绝对值，但除以 close[D] 后变成了百分比，可跨股票比较。

### 四、派生特征（静态交叉）- 4列

| 因子名 | 计算 | 说明 |
| --- | --- | --- |
| cluster_count_ratio | active_sell / (active_buy + 1) | 卖买cluster数比 |
| dist_atr_ratio | dist_sell_stop_atr / (dist_buy_stop_atr + 0.01) | 卖买ATR距离比 |
| stop_cluster_ratio_x_vol | stop_cluster_ratio * vol_zscore | 仅buy场景有效 |
| trigger_count_ratio | active_sell_cluster_count * change_pct | 卖cluster数×涨跌幅 |


### 特征汇总

| 类别 | 列数 | 来源 |
| --- | --- | --- |
| SLC静态（可比） | 12 | stop_loss_selection |
| 趋势类 | 4 | compute_dsa |
| 位置类 | 6 | compute_dsa |
| 动量类（可比） | 5+4布尔=9 | compute_bbmacd |
| 量能类 | 8 | volume_zscore |
| 风险类 | 4 | compute_dsa + 自行计算 |
| 节奏/协同类 | 6 | compute_dsa + compute_bbmacd |
| 动态特征 | 10 | stock_k_data实时计算 |
| 派生特征 | 4 | 交叉计算 |
| **合计** | **~63** |  |


## 技术栈

- 语言：Python 3
- 模型：LightGBM（回归 + 分类）
- 数据：PostgreSQL（SQLAlchemy），stop_loss_selection + stock_k_data + factor_value_1d（验证用）
- 因子计算：`features/dsa_bbmacd_24factors_viewer.py`（compute_dsa + compute_bbmacd）+ `features/volume_zscore_plotly.py`（volume_zscore）
- 评估：Spearman IC / ICIR / MAE / AUC / AP
- 可视化：matplotlib
- SHAP：shap 库

## 实现方案

### 因子计算与验证流程

```
01_build_dataset.py（独立脚本，唯一访问DB）:
  1. 读 stop_loss_selection → 获取信号列表
  2. 按股票批量从 stock_k_data 取K线（每只股票取观察期前后50天）
  3. 对每只股票调用:
     - compute_dsa(df) → 趋势+位置+节奏因子
     - compute_bbmacd(df) → 动量因子
     - volume_zscore(df) → 量能因子
     - 自行计算: atr_pct, volatility_20d, max_drawdown_60d, beta
  4. 按日期merge到观察期样本
  5. 展开观察期20天 × 计算动态特征 × 计算MFE/MAE标签
  6. 随机抽100条验证与 factor_value_1d 一致性（误差<1e-6）
  7. 保存到 output/dataset.parquet

02~05（离线运行，只读parquet）:
  直接读取 output/dataset.parquet，不访问数据库
```

### 架构映射（dsa_experiment → stop_experiment）

| dsa_experiment 7步 | stop_experiment 5步 | 核心差异 |
| --- | --- | --- |
| 01_selection + 02_build_candidate | **01_build_dataset** | 独立数据集构建脚本：读DB+实时计算因子+展开观察期→保存parquet（唯一访问DB步骤） |
| 03_train_weekly_models | **02_train_gbdt_models** | 读取离线parquet训练4个模型(sell_reg/cls + buy_reg/cls) |
| 无对应 | **03_factor_importance** | 新增：gain/split/SHAP/双模型对比+可视化 |
| 06_weekly_selector | **04_signal_selector** | 双模型评分排序+档位+veto |
| 07_daily_trading_sheet | **05_daily_trading_sheet** | 每日决策+交易清单 |


### 标签计算逻辑

```
对每条信号 s ∈ stop_loss_selection:
  对观察期每天 D = 1, 2, ..., 20:
    obs_close = 第D个交易日收盘价
    
    # 20天窗口标签（从观察日往后看）
    mfe_20 = max(high[D+1 : D+20]) / obs_close - 1
    mae_20 = min(low[D+1 : D+20]) / obs_close - 1
    
    # 分类标签
    sell_signal = (mfe_20 > 0.07)
    buy_signal  = (mae_20 < -0.03)
    
    # 涨跌停过滤
    can_buy = 非涨停板 且 obs_close > 0
```

### 模型训练设计

```python
LGB_PARAMS = {
    "num_leaves": 16, "max_depth": 5,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "seed": 42, "verbosity": -1,
}

MODELS = {
    "sell_reg": {"target": "mfe_20", "objective": "regression", "metric": "mae"},
    "sell_cls": {"target": "sell_signal", "objective": "binary", "metric": "auc"},
    "buy_reg":  {"target": "mae_20", "objective": "regression", "metric": "mae"},
    "buy_cls":  {"target": "buy_signal", "objective": "binary", "metric": "auc"},
}
```

时间序列切分：3折滚动 + 25日 embargo。

### 因子重要性分析方法

1. **gain/split 重要性**：LightGBM 内置，跨折平均
2. **跨折稳定性**：因子排名的变异系数（CV）
3. **SHAP 值**：取最后一折模型，beeswarm 图 + summary 图
4. **双模型对比**：卖点模型 vs 买点模型的因子重要性差异
5. **回归vs分类对比**：同一目标的回归/分类模型中因子重要性是否一致
6. **因子类别对比**：SLC因子 vs 趋势类 vs 位置类 vs 动量类 vs 量能类的整体贡献度

## 目录结构

```
/root/trading/stop_experiment/
├── pipeline/
│   ├── stop_config.py              # [NEW] 策略参数配置
│   ├── factor_columns.py           # [NEW] 因子列名统一定义（SSOT）+ 分类标记
│   ├── compute_factors.py          # [NEW] 调用核心计算函数，统一入口+验证
│   ├── 01_build_dataset.py         # [NEW] 独立数据集构建：读DB+计算因子+展开观察期+标签→保存parquet
│   ├── 02_train_gbdt_models.py     # [NEW] 读取离线parquet训练4个GBDT模型
│   ├── 03_factor_importance.py     # [NEW] 因子重要性分析+可视化
│   ├── 04_signal_selector.py       # [NEW] 信号精选：双模型评分+档位+veto
│   └── 05_daily_trading_sheet.py   # [NEW] 每日决策+交易清单
├── output/                          # [NEW] 输出目录（离线数据集+模型+结果）
│   ├── dataset.parquet              # 完整数据集（01输出，02~05读取）
│   ├── models/                      # 模型文件（02输出）
│   ├── fold_metrics.csv             # 各折指标（02输出）
│   ├── feature_importance.csv       # 因子重要性（02输出）
│   ├── candidate_with_scores.parquet # 带评分数据（02输出）
│   ├── factor_importance_summary.csv # 重要性汇总（03输出）
│   └── figures/                     # 可视化图表（03输出）
├── experiments/                     # [NEW] 实验脚本目录
├── backtest/                        # [NEW] 回测目录
├── EXPERIMENT_NOTES.md             # [NEW] 实验笔记
└── run_daily.py                    # [NEW] 每日生产主入口
```

### 核心文件详细说明

**pipeline/stop_config.py**

```python
OBS_DAYS = 20            # 观察期天数
EMBARGO_DAYS = 25        # 时间序列embargo天数
N_FOLDS = 3              # 滚动折数
SELL_CLS_THRESHOLD = 0.07  # 卖点分类阈值：涨幅>7%
BUY_CLS_THRESHOLD = -0.03  # 买点分类阈值：跌幅>3%
```

**pipeline/factor_columns.py**

```python
# SLC静态因子（来自stop_loss_selection，已剔除绝对值）
SLC_STATIC_COLS = [
    "sell_stop_triggered", "buy_stop_triggered",
    "active_sell_cluster_count", "active_buy_cluster_count",
    "dist_to_nearest_sell_stop_atr", "dist_to_nearest_buy_stop_atr",
    "stop_cluster_ratio", "change_pct", "vol_zscore",
    "daily_bb_width_zscore", "last_event_bars_ago", "last_event_type",
]

# 趋势类因子
TREND_COLS = ["dsa_dir", "prev_pivot_code", "trend_align_momo", "dsa_dir_age"]

# 位置类因子
POSITION_COLS = [
    "dsa_pivot_pos_01", "price_vs_dsa_vwap_pct",
    "ret_to_last_high_pct", "ret_to_last_low_pct",
    "current_pullback_from_stage_extreme_pct", "liquidity_range_pos_01",
]

# 动量类因子（只保留可比的）
MOMENTUM_COLS = [
    "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore",
    "bbmacd_sign", "bbmacd_slope_3", "bbmacd_state",
    "bbmacd_cross_upper", "bbmacd_cross_lower",
    "bbmacd_cross_up_lower", "bbmacd_cross_down_upper",
]

# 量能类因子
VOLUME_COLS = [
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10", "days_since_vol_spike",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
]

# 风险类因子
RISK_COLS = ["atr_pct", "volatility_20d", "max_drawdown_60d", "beta"]

# 节奏/协同类因子
RHYTHM_COLS = [
    "current_stage_bars", "current_stage_amp_pct", "current_stage_ret_pct",
    "prev_stage_bars", "prev_stage_amp_pct", "price_vol_coord",
]

# 动态特征（观察日实时计算）
DYNAMIC_COLS = [
    "obs_day", "ret_to_trigger", "high_to_trigger", "low_to_trigger",
    "intraday_range", "vol_ratio", "range_position", "vol_change",
    "dist_to_sell_stop_pct", "dist_to_buy_stop_pct",
]

# 派生特征
DERIVED_COLS = [
    "cluster_count_ratio", "dist_atr_ratio",
    "stop_cluster_ratio_x_vol", "trigger_count_ratio",
]

# 因子类别标记（用于分组重要性分析）
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
}

ALL_FEATURE_COLS = (
    SLC_STATIC_COLS + TREND_COLS + POSITION_COLS + MOMENTUM_COLS
    + VOLUME_COLS + RISK_COLS + RHYTHM_COLS + DYNAMIC_COLS + DERIVED_COLS
)
# ~63列
```

**pipeline/compute_factors.py**

- `compute_stock_factors(df_kline)` → 调用 compute_dsa + compute_bbmacd + volume_zscore + 自行计算风险因子
- `verify_against_db(df_factors, ts_code, n_samples=100)` → 随机抽100条与 factor_value_1d 对比验证
- 只选取 ALL_FEATURE_COLS 中需要的列，丢弃绝对值列

**pipeline/01_build_dataset.py**（独立数据集构建脚本，唯一访问DB的步骤）

- 从 stop_loss_selection 读取全部记录
- 按股票批量从 stock_k_data 取K线（每只取观察期前后50天）
- 对每只股票调用 compute_stock_factors → 得到每日因子
- 按日期merge到观察期样本
- 展开观察期20天 × 计算动态特征 × 计算MFE/MAE标签
- 涨跌停过滤：can_buy
- 验证：随机抽100条与 factor_value_1d 对比
- **保存到 output/dataset.parquet**（约29万行 × ~75列）
- 后续步骤只读 parquet，不重新构建

**pipeline/02_train_gbdt_models.py**（离线训练，只读 parquet）

- 加载 output/dataset.parquet
- 特征工程：应用 derived_features
- 3折滚动时间序列切分 + 25日 embargo
- 训练4个模型变体（sell_reg/cls + buy_reg/cls）
- 输出：models/目录、fold_metrics.csv、feature_importance.csv、candidate_with_scores.parquet

**pipeline/03_factor_importance.py**

- 读取 feature_importance.csv
- gain/split 重要性跨折平均 + 排名
- 跨折稳定性：排名变异系数
- 按因子类别汇总贡献度（SLC/趋势/位置/动量/量能/风险/节奏/动态/派生）
- SHAP 分析（最后一折模型）
- 可视化：因子重要性条形图、双模型对比图、类别贡献饼图、SHAP beeswarm图
- 输出：factor_importance_summary.csv, 图表PNG

**pipeline/04_signal_selector.py** / **pipeline/05_daily_trading_sheet.py**

- 与之前计划相同

**run_daily.py**

- 编排每日生产流程
- 步骤：信号扫描 → 候选表构建(实时计算因子+展开观察期) → 信号精选 → 每日决策

## 数据管理与离线分离原则

**核心要求**：数据集构建（耗时~分钟级）与模型训练/评估（秒~分钟级）完全分离。

```
构建阶段（01_build_dataset.py）：
  读DB → 计算因子 → 展开观察期 → 计算标签 → 保存到 stop_experiment/output/dataset.parquet
  只在数据更新时运行，不需要每次训练都重建

训练/评估阶段（02~05）：
  直接读取 output/dataset.parquet → 训练/评估/分析
  不依赖数据库连接，可离线运行
```

**离线文件结构**：

```
stop_experiment/output/
├── dataset.parquet              # 完整数据集（~29万行 × ~75列），01输出
├── models/                      # 训练好的模型文件，02输出
├── fold_metrics.csv             # 各折指标，02输出
├── feature_importance.csv       # 因子重要性，02输出
├── candidate_with_scores.parquet # 带模型评分的数据集，02输出
├── factor_importance_summary.csv # 因子重要性汇总，03输出
└── figures/                     # 可视化图表，03输出
```

## 实施注意事项

1. **数据集构建独立性**：01_build_dataset.py 是唯一访问数据库的脚本，输出 parquet 后，02~05 全部离线运行
2. **因子计算性能**：2028只股票 × 3个计算函数，估计~100秒。按股票批量查询K线避免N+1
3. **因子验证**：构建完成后随机抽100条与 factor_value_1d 对比，误差应<1e-6
4. **观察期展开性能**：14728条信号 × 20天 ≈ 29万行，动态特征向量化计算
5. **标签NaN处理**：观察期最后几天和最近的数据无法计算完整20天标签，训练时 dropna
6. **buy_stop 字段缺失**：sell场景下 buy_stop 字段非空率仅~12%，fillna(0) 处理
7. **last_event_type 编码**：label encoding
8. **未来函数检查**：观察日D的标签只用 D+1~D+20 数据，动态特征只用 D 及之前数据，因子计算只用 D 及之前K线
9. **bbmacd_slope_3**：严格来说是绝对值差分，但3日差分量级较小，暂保留观察，模型会自行判断重要性
10. **分类正负类平衡**：需检查正类比例，若严重不平衡需 scale_pos_weight
11. **因子类别分组分析**：这是本实验的核心价值——评估SLC因子 vs 传统技术因子的增量贡献