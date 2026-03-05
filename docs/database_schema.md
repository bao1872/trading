# 数据库表结构说明文档

> **重要**：每次表结构变更时必须同步更新此文档

---

## stock_div_features（背离特征表）

存储股票的背离特征数据，主键为 `(ts_code, freq, bar_time)`。

| 字段 | 类型 | 描述 |
|------|------|------|
| id | SERIAL | 自增主键 |
| ts_code | VARCHAR(20) | 股票代码（如 600547.SH） |
| freq | VARCHAR(10) | 周期（15m, 60m, d, w） |
| bar_time | TIMESTAMP | K线时间 |
| total_div_count | INTEGER | 总背离数量 |
| macd_has_div | INTEGER | MACD是否背离（0/1） |
| macd_div_type | INTEGER | MACD背离类型（0-3） |
| macd_div_len | INTEGER | MACD背离长度 |
| macd_pos_reg | INTEGER | MACD正背离数量 |
| macd_neg_reg | INTEGER | MACD负背离数量 |
| macd_pos_hid | INTEGER | MACD隐藏正背离数量 |
| macd_neg_hid | INTEGER | MACD隐藏负背离数量 |
| hist_has_div | INTEGER | HIST是否背离（0/1） |
| hist_div_type | INTEGER | HIST背离类型（0-3） |
| hist_div_len | INTEGER | HIST背离长度 |
| hist_pos_reg | INTEGER | HIST正背离数量 |
| hist_neg_reg | INTEGER | HIST负背离数量 |
| hist_pos_hid | INTEGER | HIST隐藏正背离数量 |
| hist_neg_hid | INTEGER | HIST隐藏负背离数量 |
| obv_has_div | INTEGER | OBV是否背离（0/1） |
| obv_div_type | INTEGER | OBV背离类型（0-3） |
| obv_div_len | INTEGER | OBV背离长度 |
| obv_pos_reg | INTEGER | OBV正背离数量 |
| obv_neg_reg | INTEGER | OBV负背离数量 |
| obv_pos_hid | INTEGER | OBV隐藏正背离数量 |
| obv_neg_hid | INTEGER | OBV隐藏负背离数量 |
| created_at | TIMESTAMP | 创建时间 |

**索引**：
- `idx_div_features_ts_code` ON (ts_code)
- `idx_div_features_freq` ON (freq)
- `idx_div_features_bar_time` ON (bar_time)

**唯一约束**：UNIQUE(ts_code, freq, bar_time)

---

## stock_amp_features（AMP特征表）

存储股票的AMP通道特征数据，主键为 `(ts_code, freq, bar_time)`。

| 字段 | 类型 | 描述 |
|------|------|------|
| id | SERIAL | 自增主键 |
| ts_code | VARCHAR(20) | 股票代码（如 600547.SH） |
| freq | VARCHAR(10) | 周期（15m, 60m, d, w） |
| bar_time | TIMESTAMP | K线时间 |
| window_len | INTEGER | 窗口长度 |
| final_period | INTEGER | 最终自适应周期 |
| pearson_r | FLOAT | Pearson相关系数 |
| strength_pr | FLOAT | 相关性强度 |
| bar_close | FLOAT | 收盘价 |
| bar_upper | FLOAT | 上轨价格 |
| bar_lower | FLOAT | 下轨价格 |
| close_pos_0_1 | FLOAT | 收盘价在通道内的位置（0~1） |
| activity_pos_0_1 | FLOAT | 成交量活跃度位置（0~1） |
| upper_ret_per_bar | FLOAT | 上轨每bar收益率 |
| upper_total_ret | FLOAT | 上轨总收益率 |
| lower_ret_per_bar | FLOAT | 下轨每bar收益率 |
| lower_total_ret | FLOAT | 下轨总收益率 |
| created_at | TIMESTAMP | 创建时间 |

**索引**：
- `idx_amp_features_ts_code` ON (ts_code)
- `idx_amp_features_freq` ON (freq)
- `idx_amp_features_bar_time` ON (bar_time)

**唯一约束**：UNIQUE(ts_code, freq, bar_time)

---

## 背离类型定义

| 值 | 类型 | 描述 |
|----|------|------|
| 0 | 底背离 | 价格创新低，指标未创新低 |
| 1 | 顶背离 | 价格创新高，指标未创新高 |
| 2 | 隐藏底背离 | 价格未创新低，指标创新低 |
| 3 | 隐藏顶背离 | 价格未创新高，指标创新高 |

---

## 变更历史

| 日期 | 表名 | 变更内容 |
|------|------|----------|
| 2026-02-18 | stock_div_features | 初始创建 |
| 2026-02-18 | stock_amp_features | 初始创建 |
| 2026-03-05 | stock_popularity_rank | 新增人气排名表，存储每日股票人气排名数据 |

---

## stock_popularity_rank（人气排名表）

存储每日股票的人气排名数据，主键为 `(trade_date, ts_code)`。

| 字段 | 类型 | 描述 |
|------|------|------|
| id | SERIAL | 自增主键 |
| trade_date | DATE | 交易日期 |
| ts_code | VARCHAR(20) | 股票代码（如 600547.SH） |
| name | VARCHAR(50) | 股票简称 |
| rank | INTEGER | 人气排名 |
| created_at | TIMESTAMP | 创建时间 |

**索引**：
- `idx_pop_rank_trade_date` ON (trade_date)
- `idx_pop_rank_ts_code` ON (ts_code)
- `idx_pop_rank_rank` ON (rank)

**唯一约束**：UNIQUE(trade_date, ts_code)

**写入语义**：
- 使用 UPSERT 模式（ON CONFLICT DO UPDATE）
- 冲突键：(trade_date, ts_code)
- 增量更新：根据 trade_date 判断，只更新不存在的日期或覆盖已存在日期的数据
