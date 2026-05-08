# sentiment 舆情数据抓取模块

## 用途
1. 抓取雪球网、东方财富、同花顺三个平台的股票讨论区帖子，保存到 PostgreSQL 数据库
2. 基于已抓取的帖子数据分析高质量评论博主，展示其帖子时间线

## 目录结构
```
sentiment/
├── __init__.py
├── scraper_utils.py        # 公共工具：Selenium 驱动、时间解析、安全 DOM 提取（SSOT）
├── xueqiu_scraper.py       # 雪球抓取
├── eastmoney_scraper.py    # 东方财富抓取
├── tonghuashun_scraper.py  # 同花顺抓取
├── db_operations.py        # 数据库操作封装
├── cli.py                  # 数据抓取命令行入口
├── blogger_analyzer.py     # 博主质量分析核心逻辑（频率+内容深度评分）
├── blogger_cli.py          # 博主分析与时间线展示命令行入口
└── README.md               # 本文档
```

## 数据库表结构

### stock_sentiment_posts

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | BIGSERIAL | PRIMARY KEY | 自增主键 |
| ts_code | VARCHAR(20) | NOT NULL | 股票代码 |
| post_id | VARCHAR(50) | NOT NULL | 平台帖子ID |
| author | VARCHAR(100) | | 作者名 |
| post_time | TIMESTAMP | NOT NULL | 帖子发布时间 |
| title | TEXT | | 帖子标题 |
| content | TEXT | | 帖子正文/摘要 |
| link | VARCHAR(500) | | 帖子链接 |
| source | VARCHAR(20) | DEFAULT 'xueqiu' | 来源平台 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 入库时间 |

**唯一键**: UNIQUE(ts_code, post_id, source)  
**索引**: idx_sentiment_ts_code, idx_sentiment_post_time, idx_sentiment_created_at

**写入语义**: upsert（存在则更新，不存在则插入）  
**冲突键**: (ts_code, post_id, source)

## 运行方式

### 雪球
```bash
# 全量抓取
python -m sentiment.cli --ts-code SH688615 --source xueqiu --start-time 2026-04-01

# 增量抓取
python -m sentiment.cli --ts-code SH688615 --source xueqiu --incremental
```

### 东方财富
```bash
# 全量抓取（传统分页）
python -m sentiment.cli --ts-code SH688615 --source eastmoney --start-time 2026-04-01

# 增量抓取
python -m sentiment.cli --ts-code SH688615 --source eastmoney --incremental
```

### 同花顺
```bash
# 全量抓取（滚动加载）
python -m sentiment.cli --ts-code SH688615 --source tonghuashun --start-time 2026-04-01

# 增量抓取
python -m sentiment.cli --ts-code SH688615 --source tonghuashun --incremental
```

### 三平台同时抓取
```bash
python -m sentiment.cli --ts-code SH688615 --source all --start-time 2026-04-01
```

### 批量抓取多个股票
```bash
python -m sentiment.cli --ts-code SH688615,SZ000001 --source xueqiu --start-time 2026-04-01
```

### Dry-run（仅打印不写入）
```bash
python -m sentiment.cli --ts-code SH688615 --source eastmoney --start-time 2026-05-03 --dry-run
```

### 调整参数
```bash
# 雪球/同花顺：调整滚动次数和等待时间
python -m sentiment.cli --ts-code SH688615 --source xueqiu --max-scrolls 15 --scroll-delay 3.0

# 东方财富：调整翻页数
python -m sentiment.cli --ts-code SH688615 --source eastmoney --max-pages 15
```

## 示例

```bash
# 抓取合合信息最近3天帖子（雪球）
python -m sentiment.cli --ts-code SH688615 --source xueqiu --start-time 2026-05-03

# 抓取合合信息最近3天帖子（东方财富）
python -m sentiment.cli --ts-code SH688615 --source eastmoney --start-time 2026-05-03

# 三平台同时抓取
python -m sentiment.cli --ts-code SH688615 --source all --start-time 2026-05-03
```

## 副作用
- 读写数据库表 `stock_sentiment_posts`
- 启动 Selenium WebDriver 访问各平台网站
- 可能因反爬策略触发验证码或IP限制

## 博主分析

基于已入库的帖子数据，按**发帖频率**和**内容深度**双维度加权评分，发现高质量评论博主。

### 推荐高质量博主
```bash
# 查看合合信息近3个月高质量博主 Top 10
python -m sentiment.blogger_cli recommend --ts-code SH688615 --months 3 --top-n 10

# 仅看雪球平台
python -m sentiment.blogger_cli recommend --ts-code SH688615 --months 3 --sources xueqiu
```

### 展示博主帖子时间线
```bash
# 展示某博主在合合信息下的所有帖子（按时间升序）
python -m sentiment.blogger_cli timeline --ts-code SH688615 --author "博主名" --months 3
```

### 评分算法
```
freq_score  = ln(1 + 帖子数) / ln(1 + 最大帖子数) × 100
content_score = ln(1 + 内容均长) / ln(1 + 最大内容均长) × 100
总分 = 0.6 × freq_score + 0.4 × content_score
```
频率与内容深度权重可通过 `--w-freq` / `--w-content` 调整。

## 变更记录
- 2026-05-07: 新增 blogger_analyzer.py 和 blogger_cli.py，实现高质量博主评分与时间线展示
- 2026-05-07: 删除冗余测试脚本 test_xueqiu_hehe.py 和 test_selenium.py
- 2026-05-06: 新建 sentiment 模块，新增 stock_sentiment_posts 表
- 2026-05-06: 新增东方财富与同花顺爬虫，重构提取 scraper_utils.py，冲突键改为 (ts_code, post_id, source)
