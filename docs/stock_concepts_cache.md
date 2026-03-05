# 股票概念缓存使用说明

## 功能概述

股票概念缓存功能支持同时保存到两个地方：
1. **Excel 文件**: `stock_concepts_cache.xlsx`（项目根目录）
2. **数据库表**: `stock_concepts_cache`

每次运行更新命令时，会**全量覆盖**写入数据库表，无需手动清理旧数据。

## 数据库表结构

```sql
CREATE TABLE IF NOT EXISTS stock_concepts_cache (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(50) NOT NULL,
    concepts TEXT,
    popularity_rank INTEGER,
    market_cap FLOAT,
    total_market_cap FLOAT,
    industry VARCHAR(100),
    industry_pe FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | VARCHAR(20) | 股票代码（唯一索引） |
| name | VARCHAR(50) | 股票简称 |
| concepts | TEXT | 所属概念 |
| popularity_rank | INTEGER | 人气排名 |
| market_cap | FLOAT | 流通市值 |
| total_market_cap | FLOAT | 总市值 |
| industry | VARCHAR(100) | 行业分类 |
| industry_pe | FLOAT | 行业平均 PE |
| updated_at | TIMESTAMP | 更新时间（自动） |

### 索引

- `idx_concepts_ts_code`: ts_code 索引
- `idx_concepts_name`: name 索引
- `idx_concepts_popularity`: popularity_rank 索引
- `idx_concepts_industry`: industry 索引

## 使用方法

### 1. 创建数据库表（首次使用）

```bash
python cores/cache_generator.py --create-table
```

这会创建两个表：
- `stock_popularity_rank`（人气排名表）
- `stock_concepts_cache`（股票概念缓存表）

### 2. 更新股票概念缓存（推荐）

同时保存到 Excel 和数据库：

```bash
python cores/cache_generator.py --update-cache
```

### 3. 只更新 Excel，不保存到数据库

```bash
python cores/cache_generator.py --update-cache --no-db
```

### 4. 在代码中读取数据库

```python
from app.db import get_session
from sqlalchemy import text

with get_session() as session:
    sql = """
        SELECT ts_code, name, concepts, popularity_rank, market_cap
        FROM stock_concepts_cache
        ORDER BY popularity_rank
        LIMIT 100
    """
    result = session.execute(text(sql))
    for row in result.fetchall():
        print(f"{row.ts_code} - {row.name} - 排名:{row.popularity_rank}")
```

## 数据更新策略

### Excel 文件
- 每次运行 `--update-cache` 会**完全覆盖**旧的 Excel 文件

### 数据库表
- 使用 `bulk_upsert` 函数进行**全量覆盖更新**
- 基于 `ts_code` 唯一键进行 upsert 操作
- 如果记录存在则更新，不存在则插入
- 不会删除表中已有的其他记录（因为是基于 ts_code 更新）

## 性能说明

- **数据量**: 约 5000 条股票记录
- **更新时间**: 约 10-30 秒（主要耗时在问财数据获取）
- **数据库写入**: 约 1-2 秒

## 注意事项

1. **问财 Cookie**: 需要配置有效的问财 Cookie，否则可能无法获取数据
2. **数据库连接**: 确保数据库连接配置正确（通过 `.env` 文件）
3. **唯一键冲突**: 基于 `ts_code` 进行 upsert，确保股票代码格式正确
4. **文件大小**: Excel 文件约 1-2MB，已添加到 `.gitignore`，不会被提交

## 示例输出

```
2026-03-05 16:50:45,808 - INFO - 正在通过问财获取所有 A 股的概念、人气及市值数据...
2026-03-05 16:50:48,123 - INFO - 成功获取到 5000 条原始数据，正在处理...
2026-03-05 16:50:49,456 - INFO - 正在将缓存数据写入 Excel 文件
2026-03-05 16:50:50,789 - INFO - 成功保存数据到：/Users/zhenbao/Nextcloud/coding/交易/stock_concepts_cache.xlsx
2026-03-05 16:50:50,790 - INFO - 共保存 5000 条股票数据
2026-03-05 16:50:50,791 - INFO - 正在将缓存数据写入数据库...
2026-03-05 16:50:51,234 - INFO - 写入股票概念缓存数据 5000 条
2026-03-05 16:50:51,235 - INFO - 成功保存 5000 条数据到数据库 stock_concepts_cache 表
```
