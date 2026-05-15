# backtest/ — 回测引擎与诊断工具

## 核心引擎（SSOT — Single Source of Truth）

| 文件 | 职责 | 状态 |
|---|---|---|
| `decision_core.py` | 统一决策逻辑 (买入/卖出/风控) | **SSOT** |
| `daily_state_machine.py` | 日状态推进入口 `step_day()` | **SSOT** |
| `dynamic_exit_backtest_v2.py` | `_load_data()` 共享数据加载 + `run_backtest()` 研究用壳 | **研究/Legecy** |

## 诊断工具

| 文件 | 职责 | 状态 |
|---|---|---|
| `buy_cls_diagnosis.py` | buy_cls 模型专项诊断 | 活跃 |
| `robustness_checks.py` | 阈值/成本/容量稳健性测试 | 活跃 |
| `lookahead_bias_check.py` | 前视偏差全面检查 | 活跃 |
| `performance_report.py` | 性能完整报告 | 活跃 |

## 数据生成

| 文件 | 职责 | 状态 |
|---|---|---|
| `generate_full_predictions.py` | 全量 test 集预测生成 | 活跃 |

---

## Operator Workflow — 训练发布标准命令链

```
# 1) 训练 (含自动注册)
python stop_experiment/run_training.py --auto-register

# 2) 导入 frozen predictions 到 Prediction Store
python -m stop_experiment.registries.import_frozen_predictions

# 3) 验证回测 (Prediction Store 唯一源)
python -m stop_experiment.engine.strategy_runner --mode backtest \
    --profile production --start-date 2026-03-01 --end-date 2026-05-08

# 4) 一致性门禁
python -m stop_experiment.tests_consistency.run_all_checks --ci

# 5) 晋升为 production
python -m stop_experiment.registries.promote --model mv_YYYYMMDD_retrain_v1
```

## Legacy 边界

### ✅ 正式链路 (Official — 只增强这些)

| 组件 | 路径 | 职责 |
|------|------|------|
| 统一执行引擎 | `engine/strategy_runner.py` | 唯一执行入口 |
| 版本注册表 | `registries/*.json` | 版本管理 |
| Prediction Store | `output/prediction_store/` | 预测唯一真相源 |
| 模型晋升 | `registries/promote.py` | --register / --promote |
| Frozen 导入 | `registries/import_frozen_predictions.py` | 全量导入 |

### ⚠️ 兼容链路 (Legacy — 只维护不增强)

| 组件 | 路径 | 说明 |
|------|------|------|
| 旧预测目录 | `output/predictions/` | 不再产出新数据 |
| 旧全量预测 | `output/full_test_predictions.parquet` | 参考文件，不直接读取 |
| 旧回测壳 | `dynamic_exit_backtest_v2.py` `run_backtest()` | 研究实验用 |
| 旧入口 | `pipeline/09_paper_trading_runner.py` | 已降级 wrapper |
| 旧模型路径 | `output/models_control/` | 将通过 registry 版本化 |

### ❌ 禁止再增强

- `predictions/` 直接读写 — Prediction Store 已接管
- `full_test_predictions.parquet` 直接读取 — Prediction Store 已接管
- `09_paper_trading_runner.py` 新增业务逻辑
- `stop_config.py` 新增散落常量
- 手动复制模型到 `models_control/`

---

## 已废弃（DEPRECATED）

| 文件 | 废弃原因 |
|---|---|
| `dynamic_exit_backtest.py` | v2 覆盖 |
| `buy_side_diagnosis.py` | buy_cls_diagnosis.py 覆盖 |
| `simple_backtest.py` |已被 v2 覆盖 |
| `version_benchmark.py` | 已融入 v2 |
| `environment_gating.py` | 未实际使用 |
| `label_tightening_experiment.py` | 已完成 |