# Stop-Loss Clustering 实验笔记

## 实验概述

**目标**：评估 SLC（止损聚类）信号触发后，在20天观察期内用 GBDT 模型预测买点和卖点时机的效果，以及分析各类因子的贡献度。

## 数据集

| 指标 | 值 |
|------|-----|
| 信号数 | 14,727 |
| 展开后行数 | 275,927（20天观察期） |
| 可交易+有效标签 | 262,898 |
| 股票数 | 2,027 |
| 观察日范围 | 2025-07-01 ~ 2026-05-06 |
| 特征数 | 63 |

## 模型指标 (Phase 2: 3年数据重训)

| 模型 | 目标 | 类型 | val AUC/MAE | test AUC/MAE |
|------|------|------|------------|-------------|
| sell_reg | mfe_20 | 回归 | MAE=0.0807, IC=0.6753 | MAE=0.0711, IC=0.5441 |
| sell_cls | mfe_20>7% | 分类 | AUC=0.8449, AP=0.8531 | AUC=0.7827, AP=0.7747 |
| buy_reg | mae_20 | 回归 | MAE=0.0439, IC=0.5250 | MAE=0.0511, IC=0.2963 |
| buy_cls | mae_20<-7% | 分类 | AUC=0.7632, AP=0.8778 | AUC=0.6458, AP=0.7111 |

## 因子重要性核心发现

### 卖点模型 Top 5
1. `current_stage_amp_pct` (节奏类, 30.0%)
2. `price_vs_dsa_vwap_pct` (位置类, 13.5%)
3. `prev_stage_amp_pct` (节奏类, 11.4%)
4. `ret_to_last_low_pct` (位置类, 5.6%)
5. `current_stage_ret_pct` (节奏类, 5.1%)

### 买点模型 Top 5
1. `atr_pct` (风险类, 12.9%)
2. `price_vs_dsa_vwap_pct` (位置类, 11.9%)
3. `intraday_range` (动态类, 9.6%)
4. `ret_to_last_low_pct` (位置类, 9.3%)
5. `dsa_pivot_pos_01` (位置类, 9.2%)

### 因子类别贡献度

| 类别 | sell_reg | sell_cls | buy_reg | buy_cls |
|------|----------|----------|---------|---------|
| 位置类 | 25.4% | 65.0% | 36.7% | 43.6% |
| 节奏类 | 50.7% | 16.0% | 16.0% | 16.7% |
| 风险类 | 8.8% | 5.7% | 19.0% | 12.4% |
| 动态类 | 2.1% | 3.0% | 16.9% | 13.2% |
| SLC | 2.8% | 1.5% | 1.0% | 1.8% |

### 关键发现

1. **位置类因子最重要**：`price_vs_dsa_vwap_pct` 在所有4个模型中都排名前2，是最稳定的强因子
2. **节奏类对卖点模型贡献最大**：`current_stage_amp_pct` 在 sell_reg 中贡献30%
3. **风险类对买点模型更重要**：`atr_pct` 和 `volatility_20d` 在买点模型中贡献12-19%
4. **动态特征对买点模型有增量价值**：`intraday_range` 和 `range_position` 贡献9-16%
5. **SLC 因子贡献有限**：仅1-3%，说明触发信号本身对后续涨跌预测的增量贡献有限
6. **量能类和动量类贡献较小**：均在1-6%范围

## 标签分布

| 标签 | 均值 | 标准差 | 正类比例 |
|------|------|--------|---------|
| mfe_20 | 13.3% | 16.9% | 55.9% (>7%) |
| mae_20 | -8.2% | 6.5% | 77.6% (<-3%) |

**注意**：buy_signal 正类比例77.6%偏高，意味着大部分观察日未来20天还会跌3%以上。可能需要调整阈值为 -5% 或 -7%。

## Pipeline 运行命令

```bash
# 数据集构建（唯一访问DB的步骤，~6分钟）
python stop_experiment/pipeline/01_build_dataset.py

# 模型训练（离线，~2分钟）
python stop_experiment/pipeline/02_train_gbdt_models.py

# 因子重要性分析（离线，含SHAP约5分钟）
python stop_experiment/pipeline/03_factor_importance.py

# 信号精选
python stop_experiment/pipeline/04_signal_selector.py --top-k 50

# 每日决策
python stop_experiment/pipeline/05_daily_trading_sheet.py

# 全流程（跳过已构建的步骤）
python stop_experiment/run_daily.py --skip-build --skip-train
```

## 后续优化方向

1. **buy_signal 阈值调整**：-3% 太宽松，尝试 -5% / -7%
2. **特征精简**：SLC因子贡献仅1-3%，可考虑移除
3. **超参搜索**：当前用保守默认值，可对 top 特征调优
4. **回测验证**：将模型预测信号接入回测框架验证实际收益
5. **bbmacd_slope_3**：绝对值差分因子在模型中贡献较小，可移除

---

## Phase 3 冻结与收口 (2026-05-07)

### Frozen Baseline: V1

```python
V1_BASELINE = "v1_frozen_202605"
V1_PARAMS = {
    "buy_signal_threshold": -0.07,
    "candidate_obs_days": [1, 2, 3],
    "buy_cls_exit_threshold": 0.70,
    "stop_loss": -0.07,
    "max_hold_days": 20,
    "max_stocks_default": 10,
    "strategy_default": "sell_score",
}
```

### 版本对照 (old vs new)

| 指标 | old (-5%+fixed) | new V1 (-7%+model) | Δ |
|------|-----------------|---------------------|---|
| 正类率 | 57.08% | 51.84% | -5.24% |
| AUC | 0.5646 | 0.6435 | +0.0789 |
| 校准误差 | 0.1589 | 0.1259 | -0.0330 |
| NAV | 1.5946 | 2.7354 | +1.1407 |
| 胜率 | 59.74% | 82.86% | +23.12% |
| 盈亏比 | 4.51 | 35.92 | +31.41 |

### 稳健性验证结论

- **阈值**: 9 组全正，NAV 2.30~2.74，最优 (0.70, -0.07)
- **成本**: ×1.0~×2.0 M>F 持续成立，成本不敏感
- **容量**: k=5~20 M>F 全域正，k=10 最优

### 月度切片

| 月份 | F NAV | M NAV | M-F |
|------|-------|-------|-----|
| 2026-03 | 1.273 | 1.904 | +0.631 |
| 2026-04 | 1.531 | 1.743 | +0.212 |
| 2026-05 | 1.039 | 1.039 | +0.000 |

### 代码目录

```
pipeline/   01~05: 主流程（每日运行）
backtest/
  dynamic_exit_backtest_v2.py  ✅ 当前主引擎
  version_benchmark.py         ✅ 版本验收（新建）
  robustness_checks.py         ✅ 稳健性测试（新建）
  performance_report.py        ✅ 性能报告
  buy_cls_diagnosis.py         ✅ 诊断4表
  environment_gating.py        ✅ 环境分层+月度切片
  label_tightening_experiment.py ✅ 标签实验
  generate_full_predictions.py   ✅ 预测生成
  lookahead_bias_check.py        ✅ 前视偏差检查
  simple_backtest.py             ✅ 基础函数引用源
  dynamic_exit_backtest.py     # DEPRECATED
  buy_side_diagnosis.py        # DEPRECATED
```
