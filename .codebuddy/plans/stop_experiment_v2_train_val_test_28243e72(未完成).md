---
name: stop_experiment_v2_train_val_test
overview: 基于当前实验评估，修改数据分割(train/val/test)、剔除无效特征、调整标签阈值，并实现测试集回测评估模型性能
todos:
  - id: modify-config-columns
    content: 修改 stop_config.py(阈值+分割参数) 和 factor_columns.py(剔除8列+新增2列)
    status: pending
  - id: modify-dataset
    content: 修改 01_build_dataset.py：新增has_buy_cluster+bbmacd_slope_3_pct+buy_signal阈值调整，重新构建数据集
    status: pending
    dependencies:
      - modify-config-columns
  - id: modify-train
    content: 修改 02_train_gbdt_models.py：train/val/test分割+最终模型重训，运行训练
    status: pending
    dependencies:
      - modify-dataset
  - id: modify-importance
    content: 修改 03_factor_importance.py 适配新特征，运行因子重要性分析
    status: pending
    dependencies:
      - modify-train
  - id: build-backtest
    content: 新建 backtest/simple_backtest.py，在test集上回测评估模型性能
    status: pending
    dependencies:
      - modify-train
  - id: modify-selector-sheet
    content: 修改 04+05 适配新特征，更新 EXPERIMENT_NOTES.md
    status: pending
    dependencies:
      - modify-importance
---

## 产品概述

在现有 stop_experiment 实验基础上，修复发现的问题，重新训练模型，并用测试集做回测评估模型性能。

## 核心特性

- **数据集三段分割**：训练集(2025-07~12) / 验证集(2026-01~02) / 测试集(2026-03~05)
- **剔除8个无效特征**：4个bbmacd_cross(全零) + 2个triggered(零贡献) + bbmacd_state + last_event_type(低贡献高基数)
- **归一化bbmacd_slope_3**：除以close变为百分比，使其可跨股比较
- **调整buy_signal阈值**：-3% → -5%（正类率从77.6%降至64.4%）
- **SLC缺失值处理**：buy_stop相关6列0填充 + 新增has_buy_cluster标记
- **测试集回测**：在test集上每日选top_k股票持仓，计算组合净值、超额收益、最大回撤、胜率
- **最终模型重训**：用train+val全量重训，在test上做最终评估

## 技术栈

- 沿用现有技术栈：Python 3 + LightGBM + Pandas/NumPy + Matplotlib
- 新增回测模块：基于 dsa_experiment/backtest/account_backtest.py 的模式，简化适配

## 实现方案

### 当前实验评估（6个问题）

| # | 问题 | 证据 | 影响 |
| --- | --- | --- | --- |
| 1 | buy_signal正类77.6%偏高 | mae_20<-3%阈值过宽，信息量低 | 分类模型目标不平衡，AP虚高 |
| 2 | 4个bbmacd_cross全零 | 95%+零值，4个模型gain=0 | 浪费特征位，增加噪声 |
| 3 | triggered布尔零贡献 | sell/buy_stop_triggered在3/4模型gain=0 | 无预测力 |
| 4 | 88.3%样本缺buy_stop | 仅11.7%信号有buy cluster | 6列因子大量NaN |
| 5 | bbmacd_slope_3不可比 | <10元mean=0.07, >100元mean=3.18 | 绝对值差分，引入价格量纲 |
| 6 | 无独立test集 | 3-fold CV无holdout | 无法评估真实泛化能力 |


### 修改方案

**1. 数据分割**

| 分割 | 日期范围 | 月数 | 行数(估) | 用途 |
| --- | --- | --- | --- | --- |
| train | 2025-07 ~ 2025-12 | 6 | ~177,000 | 模型训练 |
| val | 2026-01 ~ 2026-02 | 2 | ~58,000 | 早停/调参 |
| test | 2026-03 ~ 2026-05 | 2.5 | ~40,000 | 最终评估+回测 |


embargo：train与val之间25天，val与test之间25天。

**2. 特征变更**

剔除8个无效特征：

- `bbmacd_cross_upper`, `bbmacd_cross_lower`, `bbmacd_cross_up_lower`, `bbmacd_cross_down_upper`
- `sell_stop_triggered`, `buy_stop_triggered`
- `bbmacd_state`（低贡献+高基数类别，LightGBM已报negative category warning）
- `last_event_type`（低贡献+高基数类别）

新增2个特征：

- `bbmacd_slope_3_pct = bbmacd_slope_3 / close`（归一化替代）
- `has_buy_cluster`（1=有buy cluster, 0=无）

修改1个特征：

- `bbmacd_slope_3` → 替换为 `bbmacd_slope_3_pct`

净变化：63 - 8 + 2 = **57个特征**

**3. buy_signal阈值调整**

`BUY_CLS_THRESHOLD` 从 -0.03 → -0.05
正类率：77.6% → 64.4%（更平衡）

**4. 模型训练策略**

```
阶段1：CV训练（train+val，3折rolling）
  → 选超参、评估因子重要性
  
阶段2：最终模型（train+val全部，val做早停）
  → 存final模型，用于test评估

阶段3：test评估
  → IC/AUC/MAE + 回测
```

**5. 回测设计**

基于 dsa_experiment/backtest/account_backtest.py 模式简化：

- 每日从test集的精选信号中取top_k（5/10/20）股票
- T日信号 → T+1开盘买入，持N天（5/10/20）后卖出
- 交易成本：买入0.05% + 卖出0.10%（含印花税）
- 涨跌停过滤：涨停不买、跌停不卖
- 等权组合，日度换仓
- 输出：净值曲线、超额收益、最大回撤、年化收益、胜率

## 目录结构

```
/root/trading/stop_experiment/
├── pipeline/
│   ├── stop_config.py              # [MODIFY] 阈值调整+分割参数
│   ├── factor_columns.py           # [MODIFY] 剔除8列+新增2列
│   ├── compute_factors.py          # [MODIFY] bbmacd_slope_3_pct计算
│   ├── 01_build_dataset.py         # [MODIFY] has_buy_cluster+buy_signal阈值+bbmacd_slope_3_pct
│   ├── 02_train_gbdt_models.py     # [MODIFY] train/val/test分割+最终模型重训
│   ├── 03_factor_importance.py     # [MODIFY] 适配新特征
│   ├── 04_signal_selector.py       # [MODIFY] 适配新特征
│   └── 05_daily_trading_sheet.py   # [MODIFY] 适配新特征
├── backtest/
│   └── simple_backtest.py          # [NEW] test集回测
└── run_daily.py                    # [MODIFY] 适配新流程
```

## 实施注意事项

1. **01不需要重跑全量构建**：可在已有dataset.parquet基础上派生新列，只修改标签阈值和新增列
2. **bbmacd_slope_3_pct**：在01中计算 = bbmacd_slope_3 / obs_close，注意obs_close=0时置NaN
3. **has_buy_cluster**：= (active_buy_cluster_count > 0).astype(int)，NaN视为0
4. **test集严格隔离**：test集的任何统计信息不能泄露到train/val阶段
5. **回测避免未来函数**：T日信号只能用T日及之前数据，T+1开盘价执行
6. **分类正类比例变化**：buy_signal阈值改-5%后需确认各分割段的比例是否都合理
7. **回测需考虑观察期重叠**：同一信号在观察期多天出现，需去重避免重复买入