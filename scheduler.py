#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一任务调度器

功能：集中管理所有定时调度任务
作者：项目团队
修改日期：2026-04-03
版本：v1.0.0

================================================================================
当前任务列表（共 2 个任务）
================================================================================

1. 数据集构建任务 (task_数据集构建) - [已授权]
   - 函数：build_dataset_task()
   - 触发器：Cron (每天 15:01)
   - 执行时间：周一至周五 15:01（收盘后）
   - 功能：执行 python app/build_dataset.py --update
   - 日志：scheduler.log 中搜索 "数据集构建"
   - 状态：✅ 已授权并启用

2. 信号组合任务 (task_信号组合) - [已授权]
   - 函数：signals_combined_task()
   - 触发器：Cron (每天 15:20)
   - 执行时间：周一至周五 15:20（收盘后）
   - 功能：执行 python backtrader/signals_combined.py --date 当天日期 --db
   - 日志：scheduler.log 中搜索 "信号组合"
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
- 当前已授权 2 个任务：
  1. 数据集构建任务
  2. 信号组合任务

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
    
    Q2: 调度器无法启动？
    A: 检查 Python 版本（需要 3.8+），检查依赖是否安装
    
    Q3: 日志文件过大？
    A: 实施日志轮转或定期清理旧日志

调试模式：
    # 设置日志级别为 DEBUG
    export LOG_LEVEL=DEBUG
    python scheduler.py
    
    # 查看调度器详细信息
    grep -i "apscheduler" scheduler.log

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


def build_dataset_task():
    """数据集构建任务 - 已授权（收盘后执行）"""
    import subprocess
    
    logger.info("开始执行数据集构建任务")
    
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "app" / "build_dataset.py"),
            "--update",
        ]
        
        logger.info(f"执行命令：{' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 小时超时
        )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"数据集构建：{line}")
        
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"数据集构建：{line}")
        
        if result.returncode == 0:
            logger.info("✅ 数据集构建任务执行完成")
        else:
            logger.error(f"❌ 数据集构建任务执行失败，退出码：{result.returncode}")
            raise RuntimeError(f"数据集构建任务失败：{result.stderr}")
    
    except subprocess.TimeoutExpired:
        logger.error("❌ 数据集构建任务超时（超过 1 小时）")
        raise
    except Exception as e:
        logger.error(f"❌ 数据集构建任务执行失败：{e}", exc_info=True)
        raise


def signals_combined_task():
    """信号组合任务 - 已授权（收盘后执行）"""
    import subprocess
    from datetime import datetime
    
    logger.info("开始执行信号组合任务")
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "backtrader" / "signals_combined.py"),
            "--date", today,
            "--db",
        ]
        
        logger.info(f"执行命令：{' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分钟超时
        )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"信号组合：{line}")
        
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"信号组合：{line}")
        
        if result.returncode == 0:
            logger.info("✅ 信号组合任务执行完成")
        else:
            logger.error(f"❌ 信号组合任务执行失败，退出码：{result.returncode}")
            raise RuntimeError(f"信号组合任务失败：{result.stderr}")
    
    except subprocess.TimeoutExpired:
        logger.error("❌ 信号组合任务超时（超过 30 分钟）")
        raise
    except Exception as e:
        logger.error(f"❌ 信号组合任务执行失败：{e}", exc_info=True)
        raise


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("统一任务调度器启动")
    logger.info("=" * 60)
    
    # 创建调度器
    scheduler = TaskScheduler(use_async=False)
    
    # 数据集构建任务 - 每个交易日 15:01 执行（收盘后）
    scheduler.register_task(
        name='数据集构建',
        func=build_dataset_task,
        trigger='cron',
        job_id='task_数据集构建',
        hour=15,
        minute=1,
        day_of_week='mon-fri',  # 工作日
    )
    
    # 信号组合任务 - 每个交易日 15:20 执行（收盘后）
    scheduler.register_task(
        name='信号组合',
        func=signals_combined_task,
        trigger='cron',
        job_id='task_信号组合',
        hour=15,
        minute=20,
        day_of_week='mon-fri',  # 工作日
    )
    
    logger.info(f"已注册 {len(scheduler.tasks)} 个已授权任务")
    logger.info(f"任务列表：{list(scheduler.tasks.keys())}")
    
    # 启动调度器
    scheduler.start()


if __name__ == '__main__':
    main()
