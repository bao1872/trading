# 探索方案：基于DSA+ATR Rope+Bollinger因子的买入点收益风险比分析

## 一、脚本因子体系理解

### 1.1 DSA VWAP（动态摆动锚定VWAP）— 趋势结构因子
| 因子 | 含义 | 取值范围 | 买入信号直觉 |
|------|------|----------|-------------|
| `DSA_VWAP` | 锚定到swing pivot的VWAP价格线 | 价格 | 价格回归VWAP时可能是买点 |
| `DSA_DIR` | 基于近期pivot高低的方向 | 1/-1 | 方向确认 |
| `dsa_pivot_pos_01` | 在最近pivot高低点区间内的位置 | 0~1 | **低位(0~0.3)=靠近支撑，潜在买点** |
| `signed_vwap_dev_pct` | 价格偏离VWAP的百分比 | % | **负偏离大=低估，可能超卖** |
| `bull_vwap_dev_pct` | 多头方向对齐的VWAP偏离 | % | 多头趋势中的偏离 |
| `bear_vwap_dev_pct` | 空头方向对齐的VWAP偏离 | % | 空头趋势中的偏离 |
| `trend_aligned_vwap_dev_pct` | 趋势对齐的VWAP偏离（核心） | % | **负值=趋势内回调，可能买点** |
| `lh_hh_low_pos` | 在LH-HH结构中的位置 | 0~1 | 类似position_in_structure |

### 1.2 ATR Rope（ATR绳索）— 动量/突破因子
| 因子 | 含义 | 取值范围 | 买入信号直觉 |
|------|------|----------|-------------|
| `rope` | SuperTrend类指标的核心线 | 价格 | 价格接近rope时关注 |
| `rope_dir` | Rope方向 | 1/0/-1 | **方向翻转或确认时** |
| `dist_to_rope_atr` | 价格距Rope的距离（ATR归一化） | ATR倍数 | **接近0=在绳上，可能转折** |
| `rope_slope_atr_5` | 5周期Rope斜率（ATR归一化） | ATR/bar | **正值且增大=动量增强** |
| `is_consolidating` | 是否在盘整区间 | 0/1 | 盘整末期突破是经典买点 |
| `bars_since_dir_change` | 距上次方向变化的bar数 | bar数 | 刚翻转=早期信号 |
| `range_break_up` | 向上突破盘整区间 | 0/1 | **突破信号！** |
| `range_break_up_strength` | 向上突破强度（ATR归一化） | ATR | **强度越大越好** |
| `channel_pos_01` | 在通道内的位置 | 0~1 | **低位=靠近下轨，潜在买点** |
| `range_pos_01` | 在盘整区间内的位置 | 0~1 | **低位=区间下沿** |
| `rope_pivot_pos_01` | 在Rope pivot结构中的位置 | 0~1 | 类似position_in_structure |
| `range_width_atr` | 盘整区间宽度（ATR归一化） | ATR | **窄区间突破更有效** |

### 1.3 Bollinger Bands — 波动率/均值回归因子
| 因子 | 含义 | 取值范围 | 买入信号直觉 |
|------|------|----------|-------------|
| `bb_pos_01` | 在BB带内的位置 | 0~1 | **<0.2=触及下轨，超卖** |
| `bb_width_norm` | BB带宽（标准化） | % | 收缩后扩张=波动率突破 |
| `bb_width_percentile` | 宽度历史分位数 | 0~100% | **低分位= squeeze，即将突破** |
| `bb_width_change_5` | 5周期宽度变化率 | % | 正值=扩张中 |
| `bb_expanding` | BB是否在扩张 | 0/1 | 扩张初期=动量启动 |
| `bb_contracting` | BB是否在收缩 | 0/1 | 收缩末期=squeeze |
| `bb_expand_streak` | 连续扩张bar数 | bar数 | 第1~2根=早期信号 |
| `bb_contract_streak` | 连续收缩bar数 | bar数 | 长时间收缩后突破有效 |

---

## 二、探索目标与评估口径

### 2.1 核心问题
> **什么场景下作为买入点，收益风险比最大？**

### 2.2 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **ret_N** | 买入后N天收益率 | 收益能力 |
| **max_dd_N** | 买入后N天内最大回撤 | 风险度量 |
| **risk_reward_ratio** = ret_N / \|max_dd_N\| | 收益风险比 | **核心排序指标** |
| **win_rate_N** | N天后正收益概率 | 胜率 |
| **sharpe_annualized** | 年化Sharpe | 风险调整收益 |

### 2.3 收益窗口: 5日、10日、20日（主口径）、60日

### 2.4 入场方式: T0（当日收盘） / T+1（次日开盘）

---

## 三、探索方案设计（4个层次）

### 层次①：单因子截面分析
- 每个连续因子分5组(quintile)，计算每组ret_N、max_dd、胜率
- 计算IC和Rank IC，检查单调性
- **输出**: 因子重要性排名表

### 层次②：经典买入场景枚举（10个预定义场景）
| 场景ID | 名称 | 条件 | 直觉 |
|--------|------|------|------|
| S1 | Rope触底反弹 | rope_dir从-1翻转到1 且 dist_to_rope_atr < 0.5 | 超卖反弹 |
| S2 | 盘整突破 | range_break_up=1 且 range_width_atr < 中位数 | 窄幅突破 |
| S3 | BB下轨回弹 | bb_pos_01 < 0.15 且当日收阳 | 超卖反弹 |
| S4 | BB Squeeze突破 | bb_width_percentile < 20 且 bb_expanding=1 | 波动率突破 |
| S5 | DSA VWAP回归 | trend_aligned_vwap_dev_pct < -2% 且 DSA_DIR=1 | 趋势内回调 |
| S6 | 通道下轨支撑 | channel_pos_01 < 0.2 且 rope_dir=1 | 通道内回调 |
| S7 | Rope动量启动 | rope_slope_atr_5 > 0 且 rope_dir=1 且 bars_since_dir_change < 5 | 动量初期 |
| S8 | 多重确认 | S2 + 放量 | 放量突破 |
| S9 | DSA低位+Rope向上 | dsa_pivot_pos_01 < 0.4 且 rope_dir=1 | 结构低位+方向确认 |
| S10 | BB下轨+Rope支撑 | bb_pos_01 < 0.2 且 dist_to_rope_atr > -0.5 | 双重支撑 |

### 层次③：因子交互效应分析
- rope_dir × bb_pos_01 四象限
- is_consolidating × range_break_up
- dsa_pivot_pos_01 × rope_slope_atr_5
- bb_width_percentile × range_break_up

### 层次④：最优规则搜索
- 候选条件池来自层次①的高IC因子
- 枚举单因子阈值 + 双因子组合
- 过滤: 样本量≥50, 覆盖率≥1%
- 按risk_reward_ranking排序

---

## 四、实施步骤

### Step 1: 数据准备
1. 从 stock_pools 随机抽100只股票
2. 从 stock_k_data 读K线数据
3. 调用 compute_dsa() / compute_atr_rope() / compute_bollinger() 批量计算因子
4. 计算前向收益: ret_5/10/20/60 及 max_dd

### Step 2~5: 依次运行4个层次的分析

### Step 6: 综合报告

---

## 五、技术要点

### 复用原则
- 直接调用 merged_dsa_atr_rope_bb_factors.py 的三个计算函数
- 不重写任何核心计算逻辑
- 新脚本只负责: 数据获取 → 因子调用 → 分析 → 输出

### 数据源
- PostgreSQL stock_k_data 表（不用pytdx在线拉取）
- PostgreSQL stock_pools 表获取股票列表

### 性能
- 100只×800bar ≈ 80K行，计算量可控
- 全部向量化

### 风控
- 排除ST/退市股
- 考虑最小流动性过滤

### 输出
- 控制台打印核心表格
- CSV保存详细数据
