# 统一任务调度器使用手册

## 目录
- [功能说明](#功能说明)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [任务列表](#任务列表)
- [高级用法](#高级用法)
- [故障排查](#故障排查)

## 功能说明

统一任务调度器 (`scheduler.py`) 集中管理所有定时任务，包括：
- ✅ 背离扫描（每 5 分钟整点执行）
- 📊 日报生成（每天收盘后）
- 🧹 临时文件清理（每天收盘后）

### 特性
- **集中管理**：所有任务在一个地方配置和监控
- **智能调度**：支持 Cron、Interval 等多种触发方式
- **错误处理**：自动重试、错误通知、日志记录
- **优雅退出**：收到终止信号后完成当前任务再退出
- **防止并发**：同一任务不会同时运行多个实例

## 快速开始

### 1. 安装依赖

```bash
# 依赖已在 requirements.txt 中
pip install -r requirements.txt
```

### 2. 配置文件

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置（根据需要开启/关闭任务）
vim .env
```

### 3. 启动调度器

```bash
# 前台运行（调试用）
python scheduler.py

# 后台运行（生产环境）
nohup python scheduler.py > scheduler.log 2>&1 &

# 查看运行状态
ps aux | grep scheduler.py

# 查看日志
tail -f scheduler.log
```

### 4. 停止调度器

```bash
# 获取进程 ID
ps aux | grep scheduler.py

# 优雅停止（等待当前任务完成）
kill <PID>

# 强制停止
kill -9 <PID>
```

## 配置说明

### 环境变量 (.env 文件)

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `ENABLE_DIVERGENCE_SCAN` | 背离扫描任务开关 | `true` | `true`/`false` |
| `ENABLE_DAILY_REPORT` | 日报生成任务开关 | `false` | `true`/`false` |
| `ENABLE_CLEANUP` | 临时文件清理开关 | `true` | `true`/`false` |
| `DIVERGENCE_SCAN_INTERVAL` | 背离扫描间隔（分钟） | `5` | `5`, `10`, `15` |
| `LOG_LEVEL` | 日志级别 | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### 交易时段配置

背离扫描任务仅在以下时段运行：
- **上午**: 09:25 - 11:30（包含集合竞价）
- **下午**: 13:00 - 15:00
- **工作日**: 周一至周五（自动跳过节假日）

## 任务列表

### 1. 背离扫描任务

**任务名称**: `背离扫描`  
**执行时间**: 每 5 分钟整点（交易时段）  
**功能**: 扫描所有股票的背离信号并发送飞书通知

**执行逻辑**:
1. 检查是否为交易日
2. 检查是否在交易时段
3. 扫描所有股票的 1m/5m/15m/60m 周期背离
4. 发送飞书通知（仅新发现的背离）
5. 记录已推送的背离信号（去重）

**输出目录**: `背离扫描/`

### 2. 日报生成任务

**任务名称**: `日报生成`  
**执行时间**: 每天 15:30（收盘后）  
**功能**: 生成当日交易日报（待实现）

**待实现功能**:
- 当日背离信号汇总
- 涨跌幅统计
- 成交量分析
- 资金流向

### 3. 临时文件清理任务

**任务名称**: `临时文件清理`  
**执行时间**: 每天 17:00  
**功能**: 清理生成的临时文件和图片

**清理范围**:
- `背离扫描/*.png` - 临时图片文件
- `复盘/*.png` - 临时图片文件

## 高级用法

### 添加新任务

1. 在 `scheduler.py` 中定义任务函数：

```python
def my_new_task():
    """我的新任务"""
    logger.info("开始执行我的新任务")
    try:
        # 任务逻辑
        logger.info("任务执行完成")
    except Exception as e:
        logger.error(f"任务执行失败：{e}", exc_info=True)
        raise
```

2. 在 `main()` 函数中注册任务：

```python
scheduler.register_task(
    name='我的新任务',
    func=my_new_task,
    trigger='cron',  # 或 'interval'
    hour=10,         # Cron 参数
    minute=0,
)
```

### 修改任务执行时间

**Cron 表达式示例**:
```python
# 每天早上 9:25 执行
trigger='cron', hour=9, minute=25

# 每小时执行一次
trigger='cron', minute=0

# 每周一 9:00 执行
trigger='cron', hour=9, minute=0, day_of_week='mon'

# 工作日 9:25-11:30 每 5 分钟执行
trigger='cron', minute='*/5', hour='9-11', day_of_week='mon-fri'
```

**Interval 表达式示例**:
```python
# 每 5 分钟执行一次
trigger='interval', minutes=5

# 每小时执行一次
trigger='interval', hours=1

# 每天执行一次
trigger='interval', days=1
```

### 动态控制任务

```python
# 暂停任务
scheduler.scheduler.pause_job('task_背离扫描')

# 恢复任务
scheduler.scheduler.resume_job('task_背离扫描')

# 移除任务
scheduler.scheduler.remove_job('task_背离扫描')

# 立即执行任务
scheduler.scheduler.get_job('task_背离扫描').modify(next_run_time=datetime.now())
```

## 故障排查

### 1. 查看日志

```bash
# 实时查看日志
tail -f scheduler.log

# 查看最近 100 行
tail -n 100 scheduler.log

# 搜索错误日志
grep ERROR scheduler.log
```

### 2. 常见问题

**Q: 任务没有按时执行？**
- 检查 `.env` 中任务开关是否开启
- 检查当前是否在交易时段
- 查看日志是否有错误信息

**Q: 重复收到相同的背离通知？**
- 检查去重逻辑是否正常工作
- 确认 `notified_divergences` 集合是否正确维护

**Q: 调度器无法启动？**
- 检查 Python 版本（需要 3.8+）
- 检查依赖是否安装：`pip install -r requirements.txt`
- 查看日志中的错误信息

**Q: 日志文件过大？**
- 实现日志轮转（log rotation）
- 定期清理旧日志：`find . -name "*.log" -mtime +7 -delete`

### 3. 调试模式

```bash
# 设置日志级别为 DEBUG
export LOG_LEVEL=DEBUG

# 启动调度器
python scheduler.py
```

### 4. 手动触发任务

```python
# 在 Python 交互环境中
from scheduler import scan_divergence_task
scan_divergence_task()  # 手动执行背离扫描
```

## 监控与告警

### 1. 健康检查

```bash
# 检查调度器进程
ps aux | grep scheduler.py

# 检查日志更新时间
ls -l scheduler.log
```

### 2. 告警配置

在 `.env` 中配置飞书 webhook：

```bash
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
FEISHU_SECRET=xxx
```

调度器会在任务失败时发送告警通知。

## 最佳实践

1. **定期备份配置**：备份 `.env` 文件和自定义任务代码
2. **监控日志大小**：定期清理或轮转日志文件
3. **测试环境验证**：新任务先在测试环境验证
4. **设置超时时间**：防止任务长时间运行
5. **优雅退出**：使用 `kill` 而非 `kill -9` 停止调度器

## 更新日志

- **2026-03-05**: 初始版本，支持背离扫描、文件清理等任务
- 后续更新将在此记录...

## 联系方式

如有问题，请联系项目维护人员或提交 Issue。
