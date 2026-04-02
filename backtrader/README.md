# Backtrader 脚本说明文档

## 目录结构

```
backtrader/
├── signals_theme.py          # 主题投资信号生成（主脚本）
├── signals_abnormal.py        # 日线滚动放量异动统计（向量化）
├── signals_combined.py        # 合并 signals_theme + signals_abnormal 同时运行
├── trading_entry.py          # 基于CHoCH买点的回测
├── backtest_engine.py        # 基于异动信号的日线回测
├── selection_rolling.py      # 日线滚动异动选股过滤器
├── theme_mapping.json        # 主题映射配置
└── utils/
    ├── __init__.py           # 工具模块导出
    ├── concept_mapping.py     # 概念映射（股票↔概念双向映射）
    ├── theme_aggregation.py   # 主题聚合（强度计算、概念汇总）
    ├── volume_anomaly.py     # 放量异动检测（Z-Score）
    ├── limit_up_stats.py     # 涨跌停计算工具
    └── theme_mapping.py       # 主题映射加载
```

---

## 依赖关系图

```
datasource/
├── k_data_loader.py          # 行情数据加载（底层依赖）
└── database.py               # 数据库查询

signals_theme.py
├── datasource/k_data_loader.py
└── utils/
    ├── concept_mapping.py    # 加载概念数据
    └── theme_aggregation.py  # 主题聚合

signals_abnormal.py
├── datasource/k_data_loader.py
└── utils/
    ├── concept_mapping.py
    └── theme_aggregation.py

signals_combined.py
├── signals_theme.py          # 调用 generate_signals
├── signals_abnormal.py       # 调用 generate_rolling_signals
└── utils/concept_mapping.py  # 补充概念映射

trading_entry.py
└── signals_abnormal.py 的输出结果（../review/signal_*.xlsx）

backtest_engine.py
└── utils/volume_anomaly.py

selection_rolling.py
└── utils/volume_anomaly.py
```

---

## 核心脚本详解

### 1. signals_theme.py

**功能**：主题投资信号生成 - 整合放量检测+概念映射+主题聚合

**输入**：
- `datasource/k_data_loader.py` - 行情数据
- `stock_concepts_cache.xlsx` - 概念数据
- `theme_mapping.json` - 主题映射配置

**输出**：
- Excel: `../review/theme_{date}.xlsx`，包含4个Sheet：
  - 主题排名 - 按强度排序
  - 概念排名 - 按异动股数排序
  - 个股异动 - Z-Score>2.5的异动股详情
  - 涨停金字塔 - 按"几天几板"分组
- 数据库: 4张表（见下方数据库表结构）

**核心计算逻辑**：

#### 1.1 `calculate_consecutive_limit(df, max_gap=2)`

计算连板天数（涨停/跌停），核心算法：

```python
# 输入: df 含 is_limit_up_close, is_limit_down_close 列
# 输出: 添加 is_limit_up, is_limit_down, streak_count, streak_trading_days, streak_key

# 连板定义:
# - 几板 = 涨停天数（涨停日计数）
# - 几天 = 从第一板到当日的交易日数（含首尾）
# - 中间最多允许2天连续非涨停日，否则断开
```

**算法流程**（逐股票遍历）：
1. 按`ts_code`和`bar_time`排序
2. 对每只股票维护状态变量：
   - `cur_up_count` - 当前连板数
   - `last_up_idx` - 上一个涨停日索引
   - `gap_count_up` - 连续非涨停日数
   - `streak_start_i` - 当前连板起始索引
3. 遍历每个交易日：
   - 若当日涨停：计算与上次涨停的gap，gap≤2则继续累加，否则重置
   - 若当日未涨停：gap_count_up++，若>2则重置连板状态
4. 输出：`streak_key = f"{streak_days}天{streak_count}板"`

#### 1.2 `load_all_stock_data(freq, start_date, end_date)`

加载指定日期范围的行情数据：
1. 调用`iter_k_data_with_names`遍历股票
2. 计算涨跌停价格（基于前收盘价）：
   - ST/4开头：±5%
   - 688/30开头：±20%
   - 8开头：±30%
   - 普通：±10%
3. 计算`is_limit_up_close = close >= limit_up_price`
4. 调用`calculate_consecutive_limit`计算连板

#### 1.3 `aggregate_limit_up_by_theme(limit_up_df, ...)`

按主题聚合涨停股票：
1. 根据股票代码获取其所属主题
2. 按"主题 → 几天几板"分组
3. 排序：`sorted(groups.keys(), key=_streak_sort_key)`
   - 优先按板数降序，再按天数升序

---

### 2. signals_abnormal.py

**功能**：日线滚动窗口放量异动统计

**核心算法**：
1. 一次性加载全量股票数据（向量化 `groupby`）
2. 以5个交易日为窗口resample日线数据，计算总成交量
3. 用过去51个窗口计算Z-Score
4. 计算CV（变异系数）和Spearman相关系数（放量顺序）

**输入**：数据库行情数据

**输出**：
- Excel: `../review/signal_{date}.xlsx`
- 数据库: 3张表（见下方数据库表结构）

**向量化优化**：参考 `signals_theme.py`，数据加载和Z-Score计算均使用 pandas `groupby().transform()` 批量处理，替代原有的逐股票generator遍历。

---

### 3. signals_combined.py

**功能**：合并 `signals_theme.py` 和 `signals_abnormal.py` 同时运行，用一条命令输出两套结果

**核心逻辑**：
1. 调用 `signals_theme.generate_signals()` 获取主题版结果（含涨停/跌停金字塔）
2. 调用 `signals_abnormal.generate_rolling_signals()` 获取异动版结果（含CV/Spearman）
3. 合并 CV/Spearman 到主题版 `stock_anomaly`
4. 导出两套独立 Excel 文件

**输入**：数据库行情数据、概念数据、主题映射

**输出**：
- `theme_{date}.xlsx` - 主题版Excel，涨停/跌停金字塔**含概念列**（增强）
- `signal_{date}.xlsx` - 异动版Excel（与独立运行完全一致）
- 7张数据库表（theme版4张 + abnormal版3张）

**与独立运行的差异**：涨停/跌停金字塔 Sheet 比原版多一列"概念"，展示每只股票的所属概念列表。

---

### 4. trading_entry.py

**功能**：基于异动信号 + CHoCH买点的回测

**回测条件**：
1. 5日滚动窗口zscore ≥ 2.5
2. CV ≤ 0.4 或 ρ ≥ 0.8
3. CHoCH买点：异动日后60个日线bar内出现bullish CHoCH（15m或60m）
4. 买入：CHoCH出现日的下一bar开盘价（滑点0.5%）
5. 计算未来3/5/10/20/30个bar的收益统计
6. 过滤涨跌停、停牌

---

### 5. backtest_engine.py

**功能**：基于日线异动结果回测策略

**回测条件**：
1. 5日滚动窗口zscore ≥ 2.5
2. CV ≤ 0.4 或 ρ ≥ 0.8
3. 买入：第二天开盘价（滑点0.5%）
4. 计算未来3/5/10/20/30个bar的收益统计

---

### 6. selection_rolling.py

**功能**：日线滚动异动选股过滤器

**过滤条件**：
1. 5日滚动zscore ≥ 2.0
2. CV ≤ 0.4 或 Spearman ≥ 0.7

---

## 工具模块详解

### utils/concept_mapping.py

**功能**：股票代码与概念的双向映射

**函数**：
- `load_concept_data()` - 从数据库`stock_concepts_cache`表加载概念数据
- `build_stock_to_concepts(df)` - 构建`股票代码 → 概念列表`映射
- `build_concept_to_stocks(df)` - 构建`概念 → 股票列表`映射

### utils/theme_aggregation.py

**功能**：根据主题映射汇总个股异动信号

**函数**：
- `load_theme_mapping()` - 加载`theme_mapping.json`
- `build_concept_to_theme_map(mapping)` - 构建`底层概念 → 主题`映射
- `get_excluded_concepts(mapping)` - 获取需排除的概念集合
- `aggregate_theme_scores(...)` - 聚合主题强度

### utils/volume_anomaly.py

**功能**：放量异动检测 - Z-Score方法

**PERIOD_CONFIG**：
```python
{
    'daily': {'window': 51, 'freq': 'd'},
    'weekly': {'window': 9, 'freq': 'w'},
    'min60': {'window': 52*4, 'freq': '60m'},
    'min15': {'window': 42*16, 'freq': '15m'},
}
```

### utils/limit_up_stats.py

**功能**：涨跌停价格计算与连板统计

**函数**：
- `get_limit_up_price(close_price, code)` - 计算涨停价
- `get_limit_down_price(close_price, code)` - 计算跌停价
- `calculate_consecutive_details(limit_up_indices, full_indices, max_gap=4)` - 计算连板详情

---

## 数据流向

```
行情数据库
    ↓
datasource/k_data_loader.py (iter_k_data_with_names)
    ↓
signals_theme.py / signals_abnormal.py / signals_combined.py
    ↓
utils/concept_mapping.py (股票↔概念映射)
    ↓
utils/theme_aggregation.py (主题聚合)
    ↓
theme_mapping.json (主题定义)
    ↓
输出 Excel:
  - ../review/theme_{date}.xlsx  (含涨停/跌停金字塔含概念列)
  - ../review/signal_{date}.xlsx
```

---

## 使用示例

```bash
# 生成主题信号（2026-03-24）
python signals_theme.py --date 2026-03-24 --excel

# 生成异动信号
python signals_abnormal.py --date 2026-03-24

# 同时输出到Excel和数据库
python signals_theme.py --date 2026-03-24 --excel --db
python signals_abnormal.py --date 2026-03-24 --db

# 合并脚本：同时运行两个脚本，生成两套Excel + 7张数据库表
python signals_combined.py --date 2026-03-20 --db

# 运行回测
python trading_entry.py --start 2025-11-03 --end 2026-03-20

# 滚动选股过滤
python selection_rolling.py --date 2026-03-20
```

---

## 数据库表结构

使用`--db`参数时，信号结果会写入以下数据库表：

### signals_theme.py 输出表

#### 1. theme_signals（主题信号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| snapshot_date | DATE | 快照日期 |
| period | VARCHAR | 周期（daily/weekly等） |
| theme | VARCHAR | 主题名称 |
| total_zscore | FLOAT | 总Z分数 |
| avg_zscore | FLOAT | 平均Z分数 |
| stock_count | INTEGER | 股票数量 |
| strength | FLOAT | 强度 |
| anomalous_concept_count | INTEGER | 异动概念数 |
| total_concept_count | INTEGER | 总概念数 |
| concept_coverage | FLOAT | 概念覆盖率 |
| top_stocks | TEXT | TOP股票（逗号分隔） |
| top_concepts | TEXT | 贡献概念（逗号分隔） |

**唯一键**: `(snapshot_date, theme)`

#### 2. concept_signals（概念信号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| snapshot_date | DATE | 快照日期 |
| concept | VARCHAR | 概念名称 |
| theme | VARCHAR | 所属主题 |
| total_zscore | FLOAT | 总Z分数 |
| avg_zscore | FLOAT | 平均Z分数 |
| anomalous_stock_count | INTEGER | 异动股数 |
| total_stock_count | INTEGER | 成分股总数 |
| normalized_strength | FLOAT | 归一化强度 |
| intensity | FLOAT | 集中度 |
| top_stocks | TEXT | TOP股票 |

**唯一键**: `(snapshot_date, concept)`

#### 3. stock_anomaly_signals（个股异动信号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| snapshot_date | DATE | 快照日期 |
| code | VARCHAR | 股票代码 |
| name | VARCHAR | 股票名称 |
| zscore | FLOAT | Z分数 |
| date | DATE | 日期 |
| price_change | FLOAT | 涨跌幅 |
| themes_str | TEXT | 主题列表 |

**唯一键**: `(snapshot_date, code)`

#### 4. limit_up_signals（涨跌停信号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| snapshot_date | DATE | 快照日期 |
| theme | VARCHAR | 所属主题 |
| streak_key | VARCHAR | 连板标识（如"7天7板"） |
| ts_code | VARCHAR | 股票代码 |
| stock_name | VARCHAR | 股票名称 |
| streak_count | INTEGER | 连板数 |
| streak_trading_days | INTEGER | 连板天数 |
| signal_type | VARCHAR | 信号类型（limit_up/limit_down） |

**唯一键**: `(snapshot_date, ts_code, signal_type)`

### signals_abnormal.py 输出表（rolling窗口统计）

#### 5. theme_signals_rolling（rolling主题信号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| snapshot_date | DATE | 快照日期 |
| theme | VARCHAR | 主题名称 |
| total_zscore | FLOAT | 总Z分数 |
| avg_zscore | FLOAT | 平均Z分数 |
| ... | ... | 其他字段同theme_signals |

**唯一键**: `(snapshot_date, theme)`

#### 6. concept_signals_rolling（rolling概念信号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| snapshot_date | DATE | 快照日期 |
| concept | VARCHAR | 概念名称 |
| theme | VARCHAR | 所属主题 |
| ... | ... | 其他字段同concept_signals |

**唯一键**: `(snapshot_date, concept)`

#### 7. stock_anomaly_signals_rolling（rolling个股异动信号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| snapshot_date | DATE | 快照日期 |
| code | VARCHAR | 股票代码 |
| name | VARCHAR | 股票名称 |
| zscore | FLOAT | Z分数 |
| volume_cv | FLOAT | 成交量变异系数 |
| volume_spearman | FLOAT | 放量顺序相关系数 |
| date | DATE | 日期 |
| themes_str | TEXT | 主题列表 |

**唯一键**: `(snapshot_date, code)`