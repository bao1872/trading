---
name: stop_experiment_v2_backtest_diagnosis
overview: 分三层优先级：(1) 升级回测为真实交易口径（T+1买/T+N卖+成本+涨跌停+持仓去重+稳定性矩阵）；(2) buy侧专项诊断（标签漂移+分层+校准）；(3) 保留has_buy_cluster待诊断后再决定
todos:
  - id: add-cost-config
    content: 在 stop_config.py 新增 BUY_COST/SELL_COST 交易成本常量
    status: completed
  - id: rewrite-backtest
    content: 重写 simple_backtest.py 为真实持仓回测引擎：T+1 open买/T+N open卖、成本落地、涨跌停状态机、持仓去重
    status: completed
    dependencies:
      - add-cost-config
  - id: stability-matrix
    content: 实现稳定性矩阵：top_k×hold_days×strategy×成本前后 全景输出并运行
    status: completed
    dependencies:
      - rewrite-backtest
  - id: buy-diagnosis
    content: 新建 buy_side_diagnosis.py：标签漂移+分层表现+校准分析，运行诊断
    status: completed
  - id: verify-all
    content: 验证回测结果合理性+buy诊断结论汇总
    status: completed
    dependencies:
      - stability-matrix
      - buy-diagnosis
---

## 产品概述

按照三层优先级，推进 stop_experiment 实验的下一步：先做实回测口径，再做 buy 侧诊断，暂不动特征。

## 核心特性

### P1: 回测口径做实

- **真实持仓回测引擎**：T日obs_day=1信号 → T+1 open买入 → T+N open卖出
- **成本落地到成交层**：买入0.05% + 卖出0.10%（含印花税），每笔交易扣减
- **订单状态机**：涨停买不到、跌停卖不出、停牌无法交易，进入实际跳过逻辑而非过滤备注
- **持仓去重**：同一股票在持仓未结束前，不允许再次开新仓（续持不加仓）
- **稳定性矩阵**：top_k=[5,10,20] x hold_days=[3,5,10,20] x strategy=[sell_score,low_risk,composite] x cost=[gross,net]，一次性输出全景

### P2: buy侧专项诊断

- **标签漂移**：train/val/test各段buy_signal正类率、mae_20分布、分月/分价位/分行业正类率
- **分层表现**：低价/中价/高价、有无buy_cluster、不同波动率分组、不同月份的buy_cls AUC
- **校准分析**：buy_cls预测概率分桶(0.0~1.0, step=0.1) vs 真实命中率，检查单调性

### P3: 暂不删特征

- has_buy_cluster保留，等诊断后再决定是否剔除

## 技术栈

- 沿用现有：Python 3 + LightGBM + Pandas/NumPy + Matplotlib
- 回测引擎参考 dsa_experiment/backtest/account_backtest.py 的账户级模式，简化适配（等权、无分层权重、无止损）
- K线数据从 DB stock_k_data 读取

## 实现方案

### P1: 回测引擎重写

**核心思路**：参考 account_backtest.py 的 run_backtest 函数，但针对 stop_experiment 的信号结构做适配。

**信号时序**：

- obs_date 是观察日，obs_day=1 表示"该信号在观察期第一天，今日可做买入决策"
- 执行：T日(obs_date)看到obs_day=1信号 → T+1日open价买入 → 持有N个交易日 → T+N+1日open价卖出

**与 account_backtest.py 的差异**：

| 项目 | account_backtest | stop新回测 |
| --- | --- | --- |
| 信号标识 | ts_code(raw_code) | ts_code（全格式如002267.SZ） |
| 信号来源 | daily_opportunity_score | pred_sell_reg/buy_cls等4模型 |
| 选股策略 | score>分位数 | top_k + strategy |
| 权重 | 等权/分层权重 | 仅等权 |
| 止损 | 可选-5%止损 | 无止损（仅固定持仓期） |
| 稳定性矩阵 | 无 | 核心：多参数组合输出 |


**涨跌停/停牌处理**（从 account_backtest.py 提取，作为SSOT引用）：

- `is_limit_up(open, high, prev_close)`: open==high 且涨幅>=9.5%
- `is_limit_down(open, low, prev_close)`: open==low 且跌幅>=9.5%
- `is_suspended(volume)`: volume<=0
- 注意：account_backtest.py 用 raw_code（6位），stop用ts_code（9位含后缀），需要统一

**持仓去重**：holdings dict 以 ts_code 为 key，买入前检查是否已存在，已存在则跳过（续持不加仓）。

**稳定性矩阵输出**：

```
top_k = [5, 10, 20]
hold_days = [3, 5, 10, 20]  
strategy = [sell_score, low_risk, composite]
cost = [gross, net]
→ 3 × 4 × 3 × 2 = 72 种组合
```

每种组合输出：年化收益、最大回撤、Sharpe、Calmar、胜率、盈亏比、交易数。
最终输出一个 summary CSV 表格，便于对比。

### P2: buy侧诊断脚本

新建 `stop_experiment/diagnosis/buy_side_diagnosis.py`：

1. **标签漂移**：加载 dataset.parquet，按 TRAIN_END/VAL_END 分三段，统计各段 buy_signal 正类率、mae_20 的 mean/std/分位数、按月正类率变化
2. **分层表现**：按价位(低/中/高)、has_buy_cluster(0/1)、波动率分组(atr_pct分位)、月份，分别计算 buy_cls 的 AUC
3. **校准分析**：对 test 集的 buy_cls 预测概率分10桶，计算每桶真实命中率，检查单调性

关键：诊断脚本只读不写，输出到 stdout 和 CSV。

### P3: 暂不删特征

无代码改动。

## 实施注意事项

1. **回测数据流**：test_predictions.parquet 只有信号，还需要 DB K线数据做实际成交。需加载 test 期 + hold_days 余量的 K线（2026-03-01 ~ 2026-05-31）
2. **price_pivot 构建**：account_backtest 用 raw_code(6位) 做 pivot 列名，stop 用 ts_code(9位)，需要统一。建议直接用 ts_code 作为 pivot 列名
3. **信号去重**：同一 ts_code 在同一 obs_date 可能有多条记录（不同 signal_id），需 drop_duplicates(subset=[ts_code, obs_date], keep=first) 取评分最高的
4. **prev_close 计算**：涨跌停判断需要前一日收盘价，用 price_pivot 的 close.shift(1) 获取
5. **成本参数化**：BUY_COST=0.0005, SELL_COST=0.0010，在 stop_config.py 中定义，回测脚本 import
6. **buy诊断不涉及模型重训**：只读已有数据，纯分析

## 目录结构

```
stop_experiment/
├── pipeline/
│   └── stop_config.py              # [MODIFY] 新增 BUY_COST/SELL_COST 常量
├── backtest/
│   └── simple_backtest.py          # [REWRITE] 真实持仓回测引擎+稳定性矩阵
└── diagnosis/
    └── buy_side_diagnosis.py       # [NEW] buy侧专项诊断
```

### 文件详细说明

**stop_config.py** [MODIFY]

- 新增 `BUY_COST = 0.0005` 和 `SELL_COST = 0.0010`
- 用途：回测引擎和诊断脚本统一引用，禁止硬编码

**simple_backtest.py** [REWRITE]

- 重写为真实持仓回测引擎
- 核心函数 `run_account_backtest(signals, daily_prices, top_k, hold_days, strategy)` 返回 {nav_df, trades_df, summary}
- 新增 `load_daily_prices()` 从DB加载K线（参考account_backtest.py但用ts_code）
- 新增 `is_limit_up/is_limit_down/is_suspended`（从account_backtest.py引用模式，不复制）
- 新增 `run_stability_matrix()` 遍历 top_k x hold_days x strategy 输出全景表
- 保留随机baseline对比
- 输出：nav_{strategy}_k{N}_h{M}.csv, trades_{strategy}_k{N}_h{M}.csv, stability_matrix.csv

**buy_side_diagnosis.py** [NEW]

- `diagnose_label_drift(df)` → 标签漂移统计
- `diagnose_stratified_performance(df)` → 分层AUC
- `diagnose_calibration(df)` → 概率分桶校准
- 输出：diagnosis_label_drift.csv, diagnosis_stratified_auc.csv, diagnosis_calibration.csv