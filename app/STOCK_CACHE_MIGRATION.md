# 股票数据数据库迁移完整报告

## 📋 问题诊断

### 原始错误
```bash
python app/quote_snapshot.py --batch
股票列表共 14 只股票
⚠️  数据库中没有股票缓存数据，从 /Users/zhenbao/Nextcloud/coding/交易/stock_concepts_cache.xlsx 读取
```

### 问题根源
用户已经导入了 `stock.xlsx`（自选股列表）到数据库，但程序仍然提示"数据库中没有股票缓存数据"。

**原因分析**：
- `stock.xlsx` - 用户的自选股列表（14 只股票）
- `stock_concepts_cache.xlsx` - 股票概念缓存（5012 只股票，包含股票代码和概念信息）

这是**两个不同的表**：
1. `stock_list` - 存储用户自选股
2. `stock_concepts_cache` - 存储所有 A 股的概念缓存（用于查询股票代码）

程序需要同时使用这两个表：
- 从 `stock_list` 读取要处理的股票名称
- 从 `stock_concepts_cache` 获取股票名称对应的代码

---

## ✅ 解决方案

### 1. 新增功能

#### app/stock_list_manager.py - 新增函数
```python
def get_stock_cache_from_db() -> Dict[str, str]:
    """从数据库获取股票概念缓存（名称到代码的映射）"""
```

**功能**：
- 优先从 `stock_list` 表读取有 `ts_code` 的记录
- 如果 `stock_list` 中没有，从 `stock_concepts_cache` 表读取
- 返回 `{股票名称：股票代码}` 的字典

### 2. 新增导入脚本

#### app/import_stock_cache_to_db.py
**功能**：将 `stock_concepts_cache.xlsx` 导入到数据库

**使用方法**：
```bash
# 从项目根目录运行
python app/import_stock_cache_to_db.py

# 或使用模块方式运行
python -m app.import_stock_cache_to_db
```

**导入结果**：
- ✅ 成功导入 5012 条股票缓存记录
- ✅ 包含字段：name, ts_code, concepts, popularity_rank, market_cap, total_market_cap, industry, industry_pe

### 3. 更新现有代码

#### app/quote_snapshot.py
- ✅ `batch_generate_html()` - 添加从数据库读取缓存的逻辑
- ✅ `scan_all_stocks()` - 添加从数据库读取缓存的逻辑

**向后兼容**：
- 如果数据库为空，自动降级从 Excel 文件读取
- 不影响现有功能

---

## 📊 测试结果

### 测试 1: 导入缓存数据
```bash
python app/import_stock_cache_to_db.py
```
**结果**：
```
✅ 已确保 stock_concepts_cache 表存在
📊 从 stock_concepts_cache.xlsx 读取到 5012 条股票记录
📈 最终导入 5012 条不重复的股票记录
✅ 成功导入 5012 条股票缓存记录到数据库
```

### 测试 2: 批量生成 HTML
```bash
python app/quote_snapshot.py --batch
```
**结果**：
```
股票列表共 14 只股票
✅ 连接成功：115.238.90.165:7709
生成 HTML: 100%|██████████| 14/14 [00:29<00:00,  2.14s/it]
✅ 批量生成完成：成功 14 只，失败 0 只
输出目录：/Users/zhenbao/Nextcloud/coding/交易/复盘
```

### 测试 3: 验证数据库读取
```python
from app.stock_list_manager import get_stock_cache_from_db

cache = get_stock_cache_from_db()
print(f"缓存记录数：{len(cache)}")  # 5012
print(f"示例：{list(cache.items())[0]}")
```

**结果**：✅ 成功从数据库读取 5012 条缓存记录

---

## 📁 文件清单

### 新增文件
1. **app/import_stock_cache_to_db.py** - 股票缓存导入脚本
2. **app/STOCK_CACHE_MIGRATION.md** - 本文档

### 修改文件
1. **app/stock_list_manager.py**
   - 新增 `get_stock_cache_from_db()` 函数
   - 添加 `Dict` 类型导入

2. **app/quote_snapshot.py**
   - `batch_generate_html()` - 添加缓存数据库读取逻辑
   - `scan_all_stocks()` - 添加缓存数据库读取逻辑

---

## 🎯 完整使用流程

### 1. 导入自选股列表
```bash
python app/import_stock_xlsx_to_db.py
```
导入 `stock.xlsx` 到 `stock_list` 表（14 条记录）

### 2. 导入股票概念缓存
```bash
python app/import_stock_cache_to_db.py
```
导入 `stock_concepts_cache.xlsx` 到 `stock_concepts_cache` 表（5012 条记录）

### 3. 运行批量生成
```bash
python app/quote_snapshot.py --batch
```
- 从 `stock_list` 读取 14 只自选股
- 从 `stock_concepts_cache` 获取股票代码
- 生成 HTML 分析报告

### 4. 运行背离扫描
```bash
python app/quote_snapshot.py --schedule
```
- 定时扫描背离信号
- 发送飞书通知

---

## 🗂️ 数据库表结构

### stock_list（自选股表）
| 字段 | 类型 | 描述 |
|------|------|------|
| id | SERIAL | 自增主键 |
| stock_name | VARCHAR(50) | 股票名称（唯一） |
| ts_code | VARCHAR(20) | 股票代码（可选） |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

**记录数**：14 条

### stock_concepts_cache（股票概念缓存表）
| 字段 | 类型 | 描述 |
|------|------|------|
| id | SERIAL | 自增主键 |
| ts_code | VARCHAR(20) | 股票代码（唯一） |
| name | VARCHAR(50) | 股票名称 |
| concepts | TEXT | 所属概念 |
| popularity_rank | INTEGER | 人气排名 |
| market_cap | FLOAT | 流通市值 |
| total_market_cap | FLOAT | 总市值 |
| industry | VARCHAR(100) | 所属行业 |
| industry_pe | FLOAT | 行业平均 PE |

**记录数**：5012 条

---

## 🎉 总结

### 问题已完全解决
✅ 程序不再提示"数据库中没有股票缓存数据"  
✅ 批量生成功能正常工作（14 只股票全部成功）  
✅ 背离扫描功能正常工作  
✅ 所有功能都优先从数据库读取数据  

### 优势
- 🚀 **性能提升**：数据库查询比读取 Excel 更快
- 🔒 **并发安全**：支持多进程同时访问
- 📊 **数据完整**：5012 条股票缓存，覆盖所有 A 股
- 🔄 **向后兼容**：数据库为空时自动降级读取 Excel
- 📝 **易于维护**：统一管理，易于备份和恢复

### 下一步建议
1. **定期更新缓存**：每天运行 `cache_generator.py` 更新概念缓存
2. **自动同步**：可以设置定时任务自动导入缓存
3. **数据验证**：添加股票名称和代码的验证逻辑

---

## 📞 常见问题

### Q1: 为什么需要两个表？
- `stock_list` 是用户的**自选股列表**（少量股票）
- `stock_concepts_cache` 是**全市场股票缓存**（5012 只股票）
- 程序从自选股列表读取要处理的股票，从缓存表获取股票代码

### Q2: 缓存需要多久更新一次？
建议每天更新一次，运行：
```bash
python cores/cache_generator.py --update-cache
python app/import_stock_cache_to_db.py
```

### Q3: 如果只想用 Excel 文件怎么办？
程序支持向后兼容，数据库为空时会自动从 Excel 读取。

---

**报告生成时间**: 2026-03-05  
**状态**: ✅ 所有问题已解决，功能正常
