# financial_factors 脚本说明

## 目录结构

```
financial_factors/
├── db_score.py                 # 季度财务评分（权威实现，PostgreSQL 数据源）
├── batch_score.py              # 批量打分脚本（引用 db_score.py）
├── test_score_consistency.py   # 一致性验证脚本（db_score vs Tushare）
├── __init__.py                 # 包初始化
└── *.csv                      # db_score.py 的输出文件
```

## 脚本关系图

```
┌──────────────────────────────────────────────────────────────┐
│                     batch_score.py                            │
│  (批量打分脚本)                                              │
│                                                              │
│  引用 ───────────> db_score.py (核心计算逻辑)                │
│                           │                                  │
│                           ├── fetch_financial_statements_from_db() (DB)
│                           ├── prepare_base_dataframe()               │
│                           ├── add_ytd_and_ttm()                    │
│                           ├── add_factors()                        │
│                           ├── score_dataframe()                    │
│                           └── FACTOR_CONFIG (37因子)               │
│                                                              │
│  数据源 ──────────> stock_concepts_cache (股票池)         │
│                           │                                  │
│                           ▼                                  │
│                  stock_financial_score_pool (结果表)           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      db_score.py                              │
│  (可独立运行，PostgreSQL 数据源)                                  │
│                                                              │
│  数据源 ──────> stock_financial_summary (YTD累计，PostgreSQL)  │
│                                                              │
│  输出 ───────────> {ts_code}_scores.csv                    │
│                   {ts_code}_latest_summary.csv               │
│                   {ts_code}_latest_factor_scores.csv         │
└──────────────────────────────────────────────────────────────┘
```

## 数据源说明

### YTD 累计格式（stock_financial_summary）

- 数据来自 `stock_financial_summary` 表（PostgreSQL）
- 字段为 YTD 累计值，通过 `convert_flows_to_single_quarter()` 转单季度
- `convert_flows_to_single_quarter()` 支持 ytd/single 混合口径：
  - Q1 直接使用原始 YTD
  - Q2+: 当前YTD - 上一季度YTD
- 优点：无需 API 调用，可离线运行，速度快

---

## 核心函数

| 函数 | 作用 |
|------|------|
| `fetch_financial_statements_from_db(ts_code, start_date)` | 从 PostgreSQL 读取 YTD 数据并还原为单季度 |
| `prepare_base_dataframe(ts_code, start_date)` | 数据去重合并，构建基础 DataFrame |
| `add_ytd_and_ttm(df)` | 构造 YTD（年内累计）和 TTM（过去4季）衍生指标 |
| `add_factors(df)` | 构造全部 37 个财务因子 |
| `score_dataframe(df, lookback)` | 对各因子做时序分位数打分，计算维度分和总分 |
| `convert_flows_to_single_quarter(df)` | 将 YTD 累计值还原为单季度值 |
| `_convert_group_mixed_flow_to_single(g, col)` | 单股票单财年单字段的 ytd/single 混合还原 |

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

## db_score.py 使用说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ts-code` | 000657.SZ | 股票代码 |
| `--name` | 中钨高新 | 股票简称（仅输出展示用） |
| `--start` | 20120101 | 财报起始日期 YYYYMMDD |
| `--lookback` | 12 | 时序打分回看季度数（约3年） |
| `--outdir` | . | 输出目录 |

### 示例

```bash
cd /Users/zhenbao/Nextcloud/coding/交易

# 000426 兴业银锡 2025Q3 评分
python financial_factors/db_score.py \
  --ts-code 000426.SZ \
  --name "兴业银锡" \
  --start 20200101 \
  --lookback 12 \
  --outdir /tmp/score_output
```

### 输出文件

- `{ts_code}_scores.csv` — 全时间序列打分结果
- `{ts_code}_latest_summary.csv` — 最新一期总分和维度分
- `{ts_code}_latest_factor_scores.csv` — 最新一期各因子明细

---

## batch_score.py 使用说明

**用途**：对股票池（stock_concepts_cache）批量计算财务评分，结果写入数据库。

**命令行参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--start` | 20120101 | 财报起始日期 |
| `--lookback` | 12 | 时序打分回看季度数 |
| `--limit` | None | 限制股票数量（用于测试） |
| `--resume` | False | 跳过已入库股票，从断点继续 |
| `--outdir` | . | 输出目录 |

**示例**

```bash
# 批量打分（默认从 stock_concepts_cache 读取股票池）
python financial_factors/batch_score.py --resume

# 测试模式（仅3只股票）
python financial_factors/batch_score.py --limit 3
```

**数据库表**：`stock_financial_score_pool`

- 写入语义：upsert（按 ts_code + report_date）
- 重复运行会覆盖已有数据

---

## 一致性验证

`test_score_consistency.py` 以 Tushare 单季度数据为基准，验证 db_score.py 的同花顺 DB 数据质量及 YTD 还原逻辑正确性。

```bash
python financial_factors/test_score_consistency.py \
  --token "YOUR_TUSHARE_TOKEN" \
  --ts-code 000426.SZ \
  --name "兴业银锡" \
  --quarter 2025Q3
```

验证结论（000426 2025Q3）：同花顺 DB 与 Tushare 基础财务字段差异 < 0.01%，YTD 还原逻辑正确。

---

## 字段映射

| db_score 字段 | stock_financial_summary 字段 |
|---------------|-------------------------------|
| total_revenue | 营业总收入 |
| revenue | 营业收入 |
| oper_cost | 营业成本 |
| operate_profit | 营业利润 |
| ebit | EBIT |
| n_income | 归母净利润 + 少数股东损益 |
| n_income_attr_p | 归母净利润 |
| n_cashflow_act | 经营活动现金流净额 |
| capex_q | 资本开支 |
| cash_sales_q | 销售商品提供劳务收到的现金 |
| total_assets | 总资产 |
| accounts_receiv | 应收账款 |
| inventories | 存货 |
| accounts_pay | 应付账款 |
| contract_liab | 合同负债 |

---

## 注意事项

1. **数据修订问题**：当前实现按最新披露日期去重，适合当前分析。严格历史回测需按 `ann_date` 截断避免前视偏差。

2. **时序打分**：单股无法做横截面分位数，采用"相对自身历史"的分位数打分，lookback 默认 12 季度（3年）。

3. **YTD 还原**：`convert_flows_to_single_quarter()` 支持混合口径，若某季度数据缺失会导致该季度单季度值为 NaN，但不影响后续累计值计算。

4. **SSOT 原则**：`db_score.py` 是核心计算的唯一权威实现，`batch_score.py` 只做编排，不复制计算逻辑。
