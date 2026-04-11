# financial_factors 脚本说明

## 目录结构

```
financial_factors/
├── financial_quarterly_score.py  # 季度财务评分（权威实现，数据源: financial_quarterly_data）
├── sample_score.py               # 核心计算函数（prepare_base_dataframe / add_ytd_and_ttm / add_factors / score_dataframe）
├── backfill_top10_holder.py     # 股东数据回填脚本
├── top10_holder_eval_factors.py  # 股东结构因子计算
└── __init__.py                  # 包初始化
```

## 脚本关系图

```
┌──────────────────────────────────────────────────────────────┐
│              financial_quarterly_score.py                    │
│  (可独立运行，支持 single/batch 模式)                       │
│                                                              │
│  数据源 ──────> financial_quarterly_data (季频单季度值)     │
│                                                              │
│  核心计算 ────> sample_score.py                              │
│                      ├── prepare_base_dataframe()             │
│                      ├── add_ytd_and_ttm()                    │
│                      ├── add_factors()                        │
│                      └── score_dataframe()                    │
│                                                              │
│  输出 ───────────> stock_financial_score_pool (DB)          │
└──────────────────────────────────────────────────────────────┘
```

## 数据源说明

### financial_quarterly_data 表（季频单季度值）

- 数据来自 `financial_quarterly_data` 表（PostgreSQL）
- 字段为**单季度值**（非 YTD 累计值）
- 资产负债表类字段（总资产、应收、存货、应付、合同负债、股东权益）为季度末时点值
- 利润表、现金流量表字段为该季度内流量值

---

## 核心函数（sample_score.py）

| 函数 | 作用 |
|------|------|
| `prepare_base_dataframe(ts_code, start_date, ...)` | 数据读取、去重、字段标准化，构建基础 DataFrame |
| `add_ytd_and_ttm(df)` | 构造 YTD（年内累计）和 TTM（过去4季）衍生指标 |
| `add_factors(df)` | 构造全部 37 个财务因子 |
| `score_dataframe(df, lookback)` | 对各因子做时序分位数打分，计算维度分和总分 |

---

## 因子配置（FACTOR_CONFIG）

37 个因子，分 6 个维度，权重总和 1.0

| 维度 | 权重 | 核心因子 |
|------|------|----------|
| 规模与增长 | 0.24 | q_rev_yoy, q_op_yoy, q_np_parent_yoy, ytd_rev_yoy, ytd_np_parent_yoy |
| 盈利能力 | 0.18 | q_gross_margin, q_gm_yoy_change, q_op_margin, q_np_parent_margin |
| 利润质量 | 0.18 | q_cfo_to_np_parent, ttm_cfo_to_np_parent, q_accruals_to_assets |
| 现金创造能力 | 0.18 | q_cfo_to_rev, q_cfo_yoy, ytd_cfo_yoy, ttm_fcf_to_np_parent |
| 资产效率与资金占用 | 0.10 | roa_parent, cfo_to_assets |
| 边际变化与持续性 | 0.12 | q_rev_yoy_delta, q_np_parent_yoy_delta, trend_consistency |

---

## financial_quarterly_score.py 使用说明

### 单股模式

```bash
python financial_factors/financial_quarterly_score.py \
  --mode single \
  --ts-code 300857.SZ \
  --name 协创数据 \
  --start 20220101 \
  --lookback 12
```

### 批量入库模式

```bash
# 批量打分入库（默认从 stock_pools 读取股票池）
python financial_factors/financial_quarterly_score.py --mode batch --resume

# 测试模式（仅3只股票）
python financial_factors/financial_quarterly_score.py --mode batch --limit 3

# 重算指定报告期
python financial_factors/financial_quarterly_score.py --mode batch --clean 20251231
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | single | 运行模式：single 单股入库；batch 股票池批量入库 |
| `--ts-code` | - | 股票代码，single 模式下使用 |
| `--name` | - | 股票简称（仅展示用） |
| `--start` | 20120101 | 财报起始日期 |
| `--lookback` | 12 | 时序打分回看季度数（约3年） |
| `--recent-n` | 16 | 每只股票最多读取最近 N 个季度 |
| `--excel-path` | None | 可选：从 Excel 读取季频数据做单股调试 |
| `--clean` | None | 清空指定报告期后重算，如 20251231 |
| `--limit` | None | batch 模式限制股票数量（用于测试） |
| `--resume` | False | 跳过已入库股票，从断点继续 |

### 数据库表：`stock_financial_score_pool`

- 写入语义：upsert（按 ts_code + report_date）
- 重复运行会覆盖已有数据

---

## 注意事项

1. **数据修订问题**：当前实现按最新披露日期去重，适合当前分析。严格历史回测需按 `ann_date` 截断避免前视偏差。

2. **时序打分**：单股无法做横截面分位数，采用"相对自身历史"的分位数打分，lookback 默认 12 季度（3年）。

3. **SSOT 原则**：`sample_score.py` 中的 `prepare_base_dataframe / add_ytd_and_ttm / add_factors / score_dataframe` 是核心计算的唯一权威实现，`financial_quarterly_score.py` 只做编排，不复制计算逻辑。
