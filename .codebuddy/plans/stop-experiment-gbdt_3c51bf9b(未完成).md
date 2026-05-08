---
name: stop-experiment-gbdt
overview: 在根目录新建 stop_experiment 文件夹，完整复刻 dsa_experiment 的 7 步 pipeline 架构，用于 Stop-Loss Clustering 策略的 GBDT 模型训练和因子重要性评估。
todos:
  - id: create-structure
    content: 创建 stop_experiment 目录结构及基础模块（stop_config.py, factor_columns.py, derived_features.py）
    status: pending
  - id: build-01-selection
    content: 实现 01_selection_stop.py：SLC信号扫描+因子记录+写表
    status: pending
    dependencies:
      - create-structure
  - id: build-02-candidate
    content: 实现 02_build_candidate_table.py：候选池底表+多持股期标签
    status: pending
    dependencies:
      - build-01-selection
  - id: build-03-signal-model
    content: 实现 03_train_signal_models.py：信号级GBDT模型训练+因子重要性输出
    status: pending
    dependencies:
      - build-02-candidate
  - id: build-04-05-daily
    content: 实现 04_build_daily_factor_table.py 和 05_train_daily_models.py：日线因子表+日线模型
    status: pending
    dependencies:
      - build-03-signal-model
  - id: build-06-07-decision
    content: 实现 06_signal_selector.py 和 07_daily_trading_sheet.py：信号精选+每日决策
    status: pending
    dependencies:
      - build-04-05-daily
  - id: build-runner-notes
    content: 实现 run_daily.py + EXPERIMENT_NOTES.md + 自测验证
    status: pending
    dependencies:
      - build-06-07-decision
---

## 产品概述

在根目录新建 `stop_experiment` 文件夹，完整复刻 `dsa_experiment` 的 7 步 pipeline，用于 Stop-Loss Clustering（止损聚类）策略的 GBDT 实验框架。核心目标：训练 LightGBM 模型，以持股 3/5/10/20 天的收益率作为标签，评估因子重要性。

## 核心特性

- **7步完整Pipeline**：SLC信号扫描 → 候选池底表 → 信号级模型训练 → 日线因子表 → 日线模型训练 → 信号精选 → 每日决策
- **因子来源双轨制**：SLC原生因子（~25列，来自 StopLossClusteringEngine）+ factor_value表额外因子（DSA/BBMACD/量能等~42列）
- **多持股期标签**：ret_3/5/10/20_open_to_open、mae_3/5/10/20、mfe_3/5/10/20、stop_hit_5
- **SSOT原则**：SLC核心计算引用 `features/stop_loss_clustering_with_factors.py`，不重复实现
- **与dsa_experiment对齐的实验方法**：3折滚动时间序列切分 + 25日embargo + LightGBM + IC/ICIR评估

## 技术栈

- 语言：Python 3
- 模型：LightGBM（与 dsa_experiment 一致）
- 数据：PostgreSQL（SQLAlchemy），factor_value 表 + stock_k_data 表
- 因子计算：引用 `features/stop_loss_clustering_with_factors.py` 的 StopLossClusteringEngine（SSOT）
- 存储：Parquet 文件（中间数据） + PostgreSQL（触发点记录）
- 评估：Spearman IC / ICIR / AUC / MAE

## 实现方案

### 架构映射（dsa_experiment → stop_experiment）

| dsa_experiment | stop_experiment | 核心差异 |
| --- | --- | --- |
| 01_selection_dsa.py（周线BBMACD V型） | 01_selection_stop.py（日线SLC sell-stop触发） | 信号源不同：周线→日线，V型→SLC |
| 02_build_candidate_table.py | 02_build_candidate_table.py | 读取不同表，标签逻辑一致 |
| 03_train_weekly_models.py | 03_train_signal_models.py | 特征集不同：DSA因子→SLC+DSA因子 |
| 04_build_daily_factor_table.py | 04_build_daily_factor_table.py | 因子来源一致（factor_value） |
| 05_train_daily_models.py | 05_train_daily_models.py | 因子集含SLC额外特征 |
| 06_weekly_selector.py | 06_signal_selector.py | 评分维度适配SLC场景 |
| 07_daily_trading_sheet.py | 07_daily_trading_sheet.py | 决策逻辑适配SLC场景 |


### 关键技术决策

1. **01_selection_stop.py 数据来源**：重新运行 StopLossClusteringEngine 计算（用户确认），写入新表 `stop_experiment_triggers`。与 `stop_loss_selection` 表分离，因为实验需要更多因子字段。
2. **因子列组织**：`factor_columns.py` 定义 SLC_NATIVE_COLS（25列）+ EXTRA_DAILY_FACTOR_COLS（从factor_value读取的42列）+ 派生特征。
3. **标签计算**：复用 dsa_experiment 02 的 `compute_trade_labels` 逻辑（ret_3/5/10/20_open_to_open + mae + mfe + stop_hit_5），持股期从用户需求出发。
4. **模型目标**：收益模型预测 ret_5_open_to_open，风险模型预测 stop_hit_5（与dsa_experiment一致），同时评估 ret_3/10/20 的因子重要性。
5. **回补模式**：01脚本支持 --backfill 按日期范围逐日扫描全市场，每只股票立即保存（与 selection_stop.py 的 backfill 逻辑一致）。

### 性能考虑

- 01_selection_stop.py：SLC计算是CPU密集型（逐bar遍历），全市场扫描约5000只股票，需tqdm进度条 + 批量保存
- 02候选表构建：日线价格查询使用 `ANY(:codes)` 批量查询，避免N+1
- factor_value 读取：按500只一批批量查询，与dsa_experiment一致

## 目录结构

```
/root/trading/stop_experiment/
├── pipeline/
│   ├── 01_selection_stop.py        # [NEW] SLC信号扫描：运行StopLossClusteringEngine，记录全部因子+额外因子，写入stop_experiment_triggers表
│   ├── 02_build_candidate_table.py # [NEW] 候选池底表：从stop_experiment_triggers读数据，拼接日线价格+交易标签(ret_3/5/10/20,mae,mfe)
│   ├── 03_train_signal_models.py   # [NEW] 信号级模型训练：LightGBM收益+风险模型，3折滚动+25日embargo
│   ├── 04_build_daily_factor_table.py # [NEW] 日线因子表：从factor_value读取T+1~T+5日线因子快照+买卖点标签
│   ├── 05_train_daily_models.py    # [NEW] 日线模型训练：机会模型+风险模型
│   ├── 06_signal_selector.py       # [NEW] 信号精选：GBDT排序+档位+veto选股
│   ├── 07_daily_trading_sheet.py   # [NEW] 每日决策：4层决策表+交易清单
│   ├── stop_config.py              # [NEW] SLC策略参数配置
│   ├── factor_columns.py           # [NEW] 因子列名统一定义（SLC原生+额外+派生）
│   └── derived_features.py         # [NEW] 派生特征计算（SSOT）
├── experiments/                     # [NEW] 实验脚本目录（初始为空，后续追加）
├── output/                          # [NEW] 输出目录（parquet/csv/模型文件）
├── backtest/                        # [NEW] 回测目录
├── EXPERIMENT_NOTES.md             # [NEW] 实验笔记
└── run_daily.py                    # [NEW] 每日生产主入口
```

### 核心文件详细说明

**pipeline/01_selection_stop.py**

- Purpose: 重新运行 StopLossClusteringEngine，检测日线 sell-stop 触发信号，记录完整因子
- 三个入口：--ts-code（单股）、--date（全市场扫描）、--backfill（历史回补）
- 核心计算引用 `features.stop_loss_clustering_with_factors.StopLossClusteringEngine`（SSOT）
- 从 factor_value 读取额外因子（DSA/BBMACD/量能等）拼接到触发记录
- 写入 stop_experiment_triggers 表（新建，含 SLC 全部原生因子 + 额外因子）
- 过滤条件：过去5天成交额>=1亿

**pipeline/02_build_candidate_table.py**

- Purpose: 从 stop_experiment_triggers 读取触发记录，拼接日线价格路径，计算交易标签
- 标签：ret_3/5/10/20_open_to_open, mae_3/5/10/20, mfe_3/5/10/20, stop_hit_5, can_buy_next_open
- 复用 dsa_experiment/02 的 compute_trade_labels 逻辑（交易日历+批量查询）

**pipeline/03_train_signal_models.py**

- Purpose: 训练信号级收益排序模型 + 风险veto模型
- 特征：SLC_NATIVE_COLS + EXTRA_DAILY_FACTOR_COLS + 派生特征（~70列）
- 目标：ret_5_open_to_open（收益）+ stop_hit_5（风险）
- 输出：return_model/ + risk_model/ + candidate_with_scores.parquet

**pipeline/factor_columns.py**

- SLC_NATIVE_COLS：从 StopLossClusteringEngine 输出的 ~25 列因子
- EXTRA_DAILY_FACTOR_COLS：从 factor_value 读取的 ~42 列因子（与 dsa_experiment FACTOR_COLUMNS 一致）
- ALL_SIGNAL_FEATURE_COLS：SLC + EXTRA + 派生，供 03/06 使用
- DAILY_SELECT_FACTORS：日线精选因子子集，供 05/07 使用

**pipeline/stop_config.py**

- SLC 引擎参数（freq='d', model='absorbtion_extremes'）
- 实验固定参数（embargo_days=25, n_folds=3, hold_days=[3,5,10,20]）

**pipeline/derived_features.py**

- SLC场景的派生特征：如 sell_stop_scale（sum_sells_active * sell_trigger_max_vol_price）
- 交叉特征：如 cluster_ratio_x_vol_zscore, sell_dominance_x_trigger_vol
- SSOT原则：所有派生特征统一定义

## 实施注意事项

- 01_selection_stop.py 需要处理 StopLossClusteringEngine 异常（某些股票计算可能失败），与 selection_stop.py 一致使用 try/except + continue
- factor_value 中的因子可能存在缺失（某些股票/日期无记录），需 fillna(0) 处理
- 02_build_candidate_table.py 中 ts_code 格式需统一（6位 vs 带后缀），复用 dsa_experiment 的 _append_suffix 逻辑
- 日线模型（05）的 feature set 需包含信号级评分（signal_return_score, signal_risk_score），与 dsa_experiment 的 weekly_return_score/risk_score 对应
- 全市场扫描耗时长，01 脚本需支持断点续跑（按日期幂等：先删后插）