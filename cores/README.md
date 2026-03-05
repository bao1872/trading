# cores - 核心算法目录

## 目录定位

`cores/` 是核心算法库，包含所有可复用的计算逻辑（算法/公式/指标/转换/统计）。

## 核心原则

1. **单一事实源（SSOT）**：任何可复用的计算逻辑只有一个"权威实现"
2. **禁止重复（DRY）**：应用脚本只能引用调用，不得复制改写
3. **快速验证**：允许直接调用 pytdx 获取数据，满足快速验证需求
4. **可视化优先**：保留所有可视化代码，支持算法效果的直观验证

## 目录结构

```
cores/
├── amp_plotly.py             # AMP自适应移动通道算法及可视化
├── divergence_many_plotly.py # 多指标背离检测算法及可视化
├── liquidity_zones_plotly.py # 流动性区域算法及可视化
├── volume_zscore_plotly.py   # 成交量ZScore算法及可视化
├── bollinger_features_plotly.py # 布林带特征算法及可视化
├── time_features_5m_plotly.py   # 5分钟时间特征算法及可视化
├── dynamic_swing_anchored_vwap.py # 动态摆动锚定VWAP算法及可视化
├── labels_mfe_mae_score_h3_5m_plotly.py # MFE/MAE标签算法及可视化
├── smc_probability_expo_pytdx_v2.py # SMC概率指标算法及可视化
└── cache_generator.py        # 股票池缓存生成器
```

## 核心算法函数接口

### amp_plotly.py

| 函数 | 用途 | 参数 |
|------|------|------|
| `calcDevATF()` | 计算自适应周期 | df, config |
| `calc_line_value()` | 计算通道线 | df, period, config |
| `compute_activity_pos_vectorized()` | 计算成交量活跃度 | df, bar_idx, config |

### divergence_many_plotly.py

| 函数 | 用途 | 参数 |
|------|------|------|
| `compute_indicators()` | 计算技术指标 | df, config |
| `pivots_confirmed()` | 检测pivot点 | src, prd |
| `calculate_divs()` | 计算背离 | df, ind, ph_conf, pl_conf, config |

### liquidity_zones_plotly.py

| 函数 | 用途 | 参数 |
|------|------|------|
| `compute_pivots()` | 计算pivot点 | df, left_bars |
| `compute_liquidity_zones()` | 计算流动性区域 | df, config |
| `check_liquidity_grabbed()` | 检查流动性是否被抢 | df, zones |

## 使用方式

### 实验验证

```bash
# 直接运行可视化脚本
python cores/amp_plotly.py --symbol 600547 --freq d --bars 300

# 查看HTML输出
open amp.html
```

### 工程化引用

```python
# src/scanner 引用核心算法
from cores.divergence_many_plotly import (
    DivConfig, compute_indicators, calculate_divs
)

# 使用核心函数
ind = compute_indicators(df, config)
divs = calculate_divs(df, ind, ph_conf, pl_conf, config)
```

## 注意事项

1. **数据获取**：核心算法可直接调用 pytdx，但工程化模块必须通过 `k_data_loader` 获取数据
2. **输出格式**：可视化脚本输出 HTML 文件，便于快速验证
3. **修改规范**：修改核心算法时，需同步更新工程化模块的引用
