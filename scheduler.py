#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一任务调度器

功能：集中管理所有定时调度任务
作者：项目团队
修改日期：2026-03-05
版本：v1.0.0

================================================================================
当前任务列表（共 2 个任务）
================================================================================

1. 背离扫描任务 (task_背离扫描) - [已授权]
   - 函数：scan_divergence_task()
   - 触发器：Cron (每 5 分钟整点)
   - 执行时间：周一至周五 09:25-11:30, 13:00-15:00
   - 功能：扫描所有股票的 1m/5m/15m/60m 周期背离信号
   - 配置：ENABLE_DIVERGENCE_SCAN=true
   - 日志：scheduler.log 中搜索 "背离扫描"
   - 状态：✅ 已授权并启用

2. AMP 增量扫描任务 (task_amp 增量扫描) - [已授权]
   - 函数：amp_incremental_scan_task()
   - 触发器：Cron (每天 15:01)
   - 执行时间：周一至周五 15:01（收盘后）
   - 功能：增量扫描股票池的 15m/60m/d/w 周期 AMP 特征并保存到数据库
   - 配置：ENABLE_AMP_SCAN=true
   - 日志：scheduler.log 中搜索 "AMP 增量扫描"
   - 状态：✅ 已授权并启用

================================================================================
任务启停逻辑
================================================================================

启动方式：
    # 1. 前台运行（调试用，Ctrl+C 停止）
    python scheduler.py
    
    # 2. 后台运行（生产环境）
    nohup python scheduler.py > scheduler.log 2>&1 &
    
    # 3. 使用虚拟环境
    source venv/bin/activate
    python scheduler.py

停止方式：
    # 1. 优雅停止（推荐，等待当前任务完成）
    ps aux | grep scheduler.py
    kill <PID>
    
    # 2. 强制停止（不推荐，可能中断任务）
    kill -9 <PID>
    
    # 3. 使用 pkill
    pkill -f scheduler.py

查看状态：
    # 查看进程
    ps aux | grep scheduler.py
    
    # 查看监听端口（如有）
    lsof -i :<port>
    
    # 查看日志
    tail -f scheduler.log
    
    # 查看最近 100 行日志
    tail -n 100 scheduler.log

任务控制（高级）：
    # 暂停特定任务（需修改代码）
    scheduler.scheduler.pause_job('task_背离扫描')
    
    # 恢复特定任务
    scheduler.scheduler.resume_job('task_背离扫描')
    
    # 立即执行任务
    scheduler.scheduler.get_job('task_背离扫描').modify(next_run_time=datetime.now())
    
    # 移除任务
    scheduler.scheduler.remove_job('task_背离扫描')

配置管理：
    # 1. 复制示例配置
    cp .env.example .env
    
    # 2. 编辑配置
    vim .env
    
    # 3. 查看当前配置
    cat .env
    
    # 4. 热重载配置（需重启调度器）
    kill <PID> && python scheduler.py

日志管理：
    # 查看实时日志
    tail -f scheduler.log
    
    # 搜索错误日志
    grep ERROR scheduler.log
    
    # 搜索特定任务日志
    grep "背离扫描" scheduler.log
    
    # 清理旧日志（保留最近 7 天）
    find . -name "*.log" -mtime +7 -delete
    
    # 日志轮转（手动）
    mv scheduler.log scheduler.log.$(date +%Y%m%d)
    kill -USR1 <PID>  # 发送信号重新打开日志文件

================================================================================
环境变量配置 (.env 文件)
================================================================================

# 已授权任务配置
ENABLE_DIVERGENCE_SCAN=true      # 背离扫描任务（唯一已授权的任务）

# 日志配置
LOG_LEVEL=INFO                  # 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_FILE=scheduler.log         # 日志文件路径

# 通知配置（可选）
# FEISHU_WEBHOOK_URL=           # 飞书 webhook 地址
# FEISHU_SECRET=                # 飞书签名密钥

================================================================================
任务授权说明
================================================================================

根据 dispatch.md 规则第 2 条（任务授权）：
- 不允许创建或调度任何未在配置中显式指定的任务
- 所有任务必须经过明确授权才能注册到调度器
- 当前仅授权 1 个任务：背离扫描任务

如需添加新任务，必须：
1. 在本文档头部明确列出任务信息
2. 在 .env 中添加对应的配置项
3. 在 main() 函数中显式注册
4. 更新此文档说明

================================================================================
故障排查
================================================================================

常见问题：
    Q1: 任务没有按时执行？
    A: 检查 .env 中任务开关是否开启，确认当前是否在交易时段
    
    Q2: 重复收到相同的背离通知？
    A: 检查去重逻辑，确认 notified_divergences 集合是否正确维护
    
    Q3: 调度器无法启动？
    A: 检查 Python 版本（需要 3.8+），检查依赖是否安装
    
    Q4: 日志文件过大？
    A: 实施日志轮转或定期清理旧日志

调试模式：
    # 设置日志级别为 DEBUG
    export LOG_LEVEL=DEBUG
    python scheduler.py
    
    # 查看调度器详细信息
    grep -i "apscheduler" scheduler.log

手动触发任务：
    # 在 Python 交互环境中
    from scheduler import scan_divergence_task, cleanup_temp_files_task
    
    scan_divergence_task()      # 手动执行背离扫描
    cleanup_temp_files_task()   # 手动执行文件清理

================================================================================
"""
import os
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from dotenv import load_dotenv

# 加载环境变量
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 全局去重记录（在调度器生命周期内持久化）
# key: (symbol, period, type, indicator)
global_notified_divergences = set()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('scheduler')


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, use_async: bool = False):
        """初始化调度器
        
        Args:
            use_async: 是否使用异步调度器
        """
        self.use_async = use_async
        if use_async:
            self.scheduler = AsyncIOScheduler()
        else:
            self.scheduler = BlockingScheduler()
        
        # 任务注册表
        self.tasks = {}
        
        # 运行标志
        self.running = False
        
        # 注册事件监听
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        
        # 不手动注册信号处理，使用 APScheduler 的内置处理
        # APScheduler 的 BlockingScheduler 会自动处理 SIGTERM 和 SIGINT
        
        logger.info("调度器初始化完成")
    
    def _on_job_executed(self, event):
        """任务执行完成事件"""
        logger.info(f"任务 {event.job_id} 执行完成")
    
    def _on_job_error(self, event):
        """任务执行错误事件"""
        logger.error(f"任务 {event.job_id} 执行失败：{event.exception}", exc_info=event.exception)
    
    def register_task(self, name: str, func, trigger: str, job_id: str = None, **kwargs):
        """注册任务
        
        Args:
            name: 任务名称
            func: 任务函数
            trigger: 触发器类型 ('interval', 'cron', 'date')
            job_id: 可选的任务 ID（如果不提供，自动生成 task_{name}）
            **kwargs: 触发器参数
            
        Examples:
            # 每 5 分钟执行一次
            scheduler.register_task('scan_divergence', scan_func, 'interval', minutes=5)
            
            # 每天 9:25 执行
            scheduler.register_task('daily_report', report_func, 'cron', hour=9, minute=25)
            
            # 指定自定义 job_id
            scheduler.register_task('scan', func, 'cron', job_id='custom_id', hour=9)
        """
        # 提取 add_job 的参数（不属于 trigger 的参数）
        add_job_kwargs = {}
        if 'replace_existing' in kwargs:
            add_job_kwargs['replace_existing'] = kwargs.pop('replace_existing')
        if 'max_instances' in kwargs:
            add_job_kwargs['max_instances'] = kwargs.pop('max_instances')
        if 'misfire_grace_time' in kwargs:
            add_job_kwargs['misfire_grace_time'] = kwargs.pop('misfire_grace_time')
        
        if trigger == 'interval':
            trigger_obj = IntervalTrigger(**kwargs)
        elif trigger == 'cron':
            trigger_obj = CronTrigger(**kwargs)
        elif trigger == 'date':
            trigger_obj = None  # 一次性任务
        else:
            raise ValueError(f"不支持的触发器类型：{trigger}")
        
        # 使用提供的 job_id 或自动生成
        final_job_id = job_id if job_id else f"task_{name}"
        
        if trigger_obj:
            self.scheduler.add_job(
                func=func,
                trigger=trigger_obj,
                id=final_job_id,
                name=name,
                replace_existing=True,
                max_instances=1,  # 防止并发
                misfire_grace_time=60,  # 错过执行时间的容忍度
                **add_job_kwargs  # 额外的参数
            )
        else:
            self.scheduler.add_job(
                func=func,
                trigger='date',
                id=final_job_id,
                name=name,
                run_date=kwargs.get('run_date'),
            )
        
        self.tasks[name] = final_job_id
        logger.info(f"注册任务：{name} (id={final_job_id}, trigger={trigger}, params={kwargs})")
    
    def start(self):
        """启动调度器"""
        self.running = True
        logger.info("启动调度器...")
        
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("收到中断信号，停止调度器")
        finally:
            self.shutdown(wait=True)
    
    def shutdown(self, wait: bool = True):
        """关闭调度器
        
        Args:
            wait: 是否等待当前任务完成
        """
        logger.info("关闭调度器...")
        self.running = False
        
        # 关闭调度器（会等待当前任务完成）
        self.scheduler.shutdown(wait=wait)
        
        logger.info("调度器已关闭")


def scan_divergence_task():
    """背离扫描任务 - 已授权（定时扫描模式，不生成文件）"""
    from app.quote_snapshot import scan_all_stocks
    from scheduler import global_notified_divergences
    
    logger.info("开始执行背离扫描任务")
    
    # 股票列表现在从数据库读取，stock_list_path 参数仅用于向后兼容
    stock_list_path = PROJECT_ROOT / 'stock.xlsx'
    stock_cache_path = PROJECT_ROOT / 'stock_concepts_cache.xlsx'
    output_dir = PROJECT_ROOT / '背离扫描'
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 执行扫描（定时扫描模式：generate_files=False，不生成任何文件）
        # 使用全局去重记录，避免重复推送
        result = scan_all_stocks(
            stock_list_path=str(stock_list_path),
            stock_cache_path=str(stock_cache_path),
            output_dir=str(output_dir),
            notify=True,
            notified_divergences=global_notified_divergences,  # 使用全局去重记录
            generate_files=False,  # 定时扫描不生成文件
        )
        
        if result:
            logger.info(f"背离扫描完成，发现 {len(result)} 只有背离信号的股票")
        else:
            logger.info("背离扫描完成，未发现背离信号")
    
    except Exception as e:
        logger.error(f"背离扫描任务执行失败：{e}", exc_info=True)
        raise


def amp_incremental_scan_task():
    """AMP 增量扫描任务 - 已授权（收盘后执行）"""
    import subprocess
    
    logger.info("开始执行 AMP 增量扫描任务")
    
    try:
        # 调用 amp_scanner.py 进行增量扫描
        # 扫描周期：15m, 60m, d, w
        cmd = [
            sys.executable,  # 使用当前 Python 解释器
            "-m", "app.amp_scanner",
            "--freqs", "15m,60m,d,w",
        ]
        
        logger.info(f"执行命令：{' '.join(cmd)}")
        
        # 执行命令并捕获输出
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 小时超时
        )
        
        # 记录输出
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"AMP 扫描：{line}")
        
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"AMP 扫描：{line}")
        
        # 检查执行结果
        if result.returncode == 0:
            logger.info("✅ AMP 增量扫描任务执行完成")
        else:
            logger.error(f"❌ AMP 增量扫描任务执行失败，退出码：{result.returncode}")
            raise RuntimeError(f"AMP 扫描失败：{result.stderr}")
    
    except subprocess.TimeoutExpired:
        logger.error("❌ AMP 增量扫描任务超时（超过 2 小时）")
        raise
    except Exception as e:
        logger.error(f"❌ AMP 增量扫描任务执行失败：{e}", exc_info=True)
        raise


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("统一任务调度器启动")
    logger.info("=" * 60)
    
    # 创建调度器
    scheduler = TaskScheduler(use_async=False)
    
    # 从环境变量读取任务配置（仅支持已授权的任务）
    ENABLE_DIVERGENCE_SCAN = os.getenv('ENABLE_DIVERGENCE_SCAN', 'true').lower() == 'true'
    ENABLE_AMP_SCAN = os.getenv('ENABLE_AMP_SCAN', 'true').lower() == 'true'
    
    # 背离扫描任务 - 每 5 分钟整点执行（仅在 A 股交易时段）
    # 交易时段：9:25-11:30（上午），13:00-15:00（下午）
    # 注意：9:25 是集合竞价结束时间，9:30 开始连续竞价
    if ENABLE_DIVERGENCE_SCAN:
        # 上午：9 点 (25-55 分), 10 点 (0-55 分), 11 点 (0-30 分)
        scheduler.register_task(
            name='背离扫描',
            func=scan_divergence_task,
            trigger='cron',
            minute='25,30,35,40,45,50,55',  # 9 点从 25 分开始
            hour='9',  # 9 点
            day_of_week='mon-fri',  # 工作日
            id='task_背离扫描_am',
            replace_existing=True,
        )
        scheduler.register_task(
            name='背离扫描',
            func=scan_divergence_task,
            trigger='cron',
            minute='*/5',  # 每 5 分钟
            hour='10',  # 10 点整小时
            day_of_week='mon-fri',  # 工作日
            id='task_背离扫描_am2',
            replace_existing=True,
        )
        scheduler.register_task(
            name='背离扫描',
            func=scan_divergence_task,
            trigger='cron',
            minute='0,5,10,15,20,25,30',  # 11 点到 30 分结束
            hour='11',  # 11 点
            day_of_week='mon-fri',  # 工作日
            id='task_背离扫描_am3',
            replace_existing=True,
        )
        # 下午：13 点 (0-55 分), 14 点 (0-55 分), 15 点 (0 分)
        scheduler.register_task(
            name='背离扫描',
            func=scan_divergence_task,
            trigger='cron',
            minute='*/5',  # 每 5 分钟
            hour='13-14',  # 13-14 点
            day_of_week='mon-fri',  # 工作日
            id='task_背离扫描_pm',
            replace_existing=True,
        )
        scheduler.register_task(
            name='背离扫描',
            func=scan_divergence_task,
            trigger='cron',
            minute='0',  # 15 点只在 0 分执行（收盘）
            hour='15',  # 15 点
            day_of_week='mon-fri',  # 工作日
            id='task_背离扫描_pm2',
            replace_existing=True,
        )
    else:
        logger.warning("背离扫描任务未启用（ENABLE_DIVERGENCE_SCAN=false）")
        logger.warning("调度器中没有其他已授权的任务，将不会执行任何任务")
    
    # AMP 增量扫描任务 - 每天 15:01 执行（收盘后）
    if ENABLE_AMP_SCAN:
        scheduler.register_task(
            name='AMP 增量扫描',
            func=amp_incremental_scan_task,
            trigger='cron',
            hour=15,
            minute=1,
            day_of_week='mon-fri',  # 工作日
        )
    else:
        logger.warning("AMP 增量扫描任务未启用（ENABLE_AMP_SCAN=false）")
    
    logger.info(f"已注册 {len(scheduler.tasks)} 个已授权任务")
    logger.info(f"任务列表：{list(scheduler.tasks.keys())}")
    
    # 启动调度器
    scheduler.start()


if __name__ == '__main__':
    main()
