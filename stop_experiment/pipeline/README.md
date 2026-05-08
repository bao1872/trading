# pipeline/ — 核心流水线（训练 + 生产）

## 训练流水线（离线，一次性）

统一入口: `../run_training.py`

| 文件 | 职责 | 上游 | 下游 |
|---|---|---|---|
| `01_build_dataset.py` | 数据集构建（唯一访问 DB 的步骤） | DB: stop_loss_selection, stock_k_data | `02_train_gbdt_models.py` |
| `02_train_gbdt_models.py` | GBDT 模型训练（4 个变体: sell_reg/sell_cls/buy_reg/buy_cls） | `01_build_dataset.py` | `03_factor_importance.py`, `04_signal_selector.py` |
| `03_factor_importance.py` | 因子重要性分析与可视化 | `02_train_gbdt_models.py` | — |
| `04_signal_selector.py` | 信号精选排序（综合 4 模型评分） | `02_train_gbdt_models.py` | `generate_full_predictions.py` (在 backtest/) |

## 生产流水线（每日，重复）

统一入口: `../run_daily.py`

| 文件 | 职责 | 上游 | 下游 |
|---|---|---|---|
| `06_daily_inference_replay.py` | 日级推理回放 + 回测对账（验证决策一致性） | `full_test_predictions.parquet` | — |
| `08_daily_inference_report.py` | 日级推理预测报告（含双轨仓位映射 W1/W3） | `full_test_predictions.parquet`, DB | — |

## 共享配置与工具

| 文件 | 职责 |
|---|---|
| `stop_config.py` | 全局参数配置（观察期、成本、阈值、数据分割点） |
| `compute_factors.py` | 因子计算工具（~80 个因子的计算逻辑） |
| `factor_columns.py` | 因子列定义与分类 |
| `__init__.py` | 包初始化 |

## 已废弃

| 文件 | 说明 |
|---|---|
| `05_daily_trading_sheet.py` | 旧版交易清单生成，被 `08_daily_inference_report.py` 替代 |

## 踩坑记录

1. **导入 06_daily_inference_replay**: Python 不能 import 数字开头的文件名 → 使用 `importlib.util.spec_from_file_location`
2. **sold/held 优先级**: `held_codes` 检查在 `sold_codes` 之前 → 移动 `sold_codes` 检查到前面
3. **混合类型排序**: `obs_day_counts` 含 int 键和 "?" 字符串键 → 手动排序替代 `pd.Series.sort_index`
