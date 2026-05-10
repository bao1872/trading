# backtest/ — 回测引擎与诊断工具

## 核心引擎（SSOT — Single Source of Truth）

| 文件 | 职责 | 状态 |
|---|---|---|
| `dynamic_exit_backtest_v2.py` | 动态退出回测引擎（当前主引擎） | **活跃** |

**引擎特性**:
- 退出模式: model_exit（pred_buy_cls > 0.70）+ stop_loss（-7%）+ max_hold（20d）
- 买入: T 日信号 → T+1 开盘买入，等权分配
- 成本: 买入 0.05% + 卖出 0.10%
- 候选池: obs_day=1（生产口径，经聚合实验验证最优）

## 诊断工具

| 文件 | 职责 | 状态 |
|---|---|---|
| `buy_cls_diagnosis.py` | buy_cls 模型专项诊断（标签漂移/校准/分层） | 活跃 |
| `robustness_checks.py` | 阈值/成本/容量稳健性测试 | 活跃 |
| `lookahead_bias_check.py` | 前视偏差全面检查 | 活跃 |
| `performance_report.py` | 性能完整报告（胜率/夏普/Calmar 等） | 活跃 |

## 数据生成

| 文件 | 职责 | 状态 |
|---|---|---|
| `generate_full_predictions.py` | 全量 test 集预测生成（final 模型） | 活跃 |

## 已废弃（DEPRECATED）

| 文件 | 废弃原因 | 替代文件 |
|---|---|---|
| `dynamic_exit_backtest.py` | v2 移除 sell_reg exit（已被证伪无边际贡献），改为 buy-only exit | `dynamic_exit_backtest_v2.py` |
| `buy_side_diagnosis.py` | 核心诊断能力已由 buy_cls_diagnosis.py 覆盖 | `buy_cls_diagnosis.py` |
| `simple_backtest.py` | 早期 baseline，固定持有期回测，已被 v2 覆盖 | `dynamic_exit_backtest_v2.py` |
| `version_benchmark.py` | 版本对比实验，已融入 v2 | `dynamic_exit_backtest_v2.py` |
| `environment_gating.py` | 环境门控实验，未实际使用 | — |
| `label_tightening_experiment.py` | 标签收紧实验（-0.05 → -0.07），已完成 | — |
