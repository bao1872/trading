#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一任务调度器

功能：集中管理所有定时调度任务
作者：项目团队
修改日期：2026-04-14
版本：v1.2.0

================================================================================
当前任务列表（共 2 个任务）
================================================================================

1. 每日数据更新与选股任务 (task_每日数据更新与选股) - [已授权]
   - 函数：daily_update_and_selection_task()
   - 触发器：Cron (每天 16:00)
   - 执行时间：周一至周五 16:00（收盘后）
   - 执行流程：
     1. 日线数据增量更新（每天）
     2. 周线数据增量更新（仅周五）
     3. 执行选股任务（每天）
   - 配置：ENABLE_DAILY_WORKFLOW=true
   - 日志：scheduler.log 中搜索 "每日数据更新与选股"
   - 状态：✅ 已授权并启用

2. Streamlit 可视化应用启动 - [已授权]
   - 应用：vis/financial_score_app.py
   - 端口：8501
   - 访问地址：http://localhost:8501
   - 配置：ENABLE_STREAMLIT=true
   - 状态：✅ 已授权并启用（脚本启动时自动启动）

================================================================================
系统服务管理（重要）
================================================================================

本脚本已配置为 systemd 服务，开机自动启动。修改本文件后必须重启服务才能生效。

服务名称：trading-scheduler.service

【修改文件后必须执行】
    # 重新加载配置并重启服务
    systemctl daemon-reload && systemctl restart trading-scheduler.service

【常用服务命令】
    # 查看服务状态
    systemctl status trading-scheduler.service
    
    # 启动服务
    systemctl start trading-scheduler.service
    
    # 停止服务
    systemctl stop trading-scheduler.service
    
    # 重启服务
    systemctl restart trading-scheduler.service
    
    # 启用开机自启
    systemctl enable trading-scheduler.service
    
    # 禁用开机自启
    systemctl disable trading-scheduler.service
    
    # 查看是否开机自启
    systemctl is-enabled trading-scheduler.service

【查看服务日志】
    # 实时查看日志
    journalctl -u trading-scheduler.service -f
    
    # 查看最近 100 行日志
    journalctl -u trading-scheduler.service -n 100 --no-pager
    
    # 查看今天的日志
    journalctl -u trading-scheduler.service --since today --no-pager

【服务配置文件位置】
    /etc/systemd/system/trading-scheduler.service

================================================================================
任务启停逻辑（手动运行方式，不推荐用于生产环境）
================================================================================

启动方式：
    # 1. 前台运行（调试用，Ctrl+C 停止）
    python scheduler.py
    
    # 2. 后台运行（生产环境，不推荐）
    nohup python scheduler.py > scheduler.log 2>&1 &
    
    # 3. 使用虚拟环境
    source .venv311/bin/activate
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
    
    # 查看日志
    tail -f scheduler.log
    
    # 查看最近 100 行日志
    tail -n 100 scheduler.log

任务控制（高级）：
    # 暂停特定任务
    scheduler.scheduler.pause_job('task_每日数据更新与选股')
    
    # 恢复特定任务
    scheduler.scheduler.resume_job('task_每日数据更新与选股')
    
    # 立即执行任务
    scheduler.scheduler.get_job('task_每日数据更新与选股').modify(next_run_time=datetime.now())
    
    # 移除任务
    scheduler.scheduler.remove_job('task_每日数据更新与选股')

配置管理：
    # 1. 复制示例配置
    cp .env.example .env
    
    # 2. 编辑配置
    vim .env
    
    # 3. 查看当前配置
    cat .env
    
    # 4. 热重载配置（需重启服务）
    systemctl restart trading-scheduler.service

日志管理：
    # 查看实时日志（推荐，使用 journalctl）
    journalctl -u trading-scheduler.service -f
    
    # 搜索错误日志
    journalctl -u trading-scheduler.service --since today | grep ERROR
    
    # 搜索特定任务日志
    journalctl -u trading-scheduler.service --since today | grep "每日数据更新与选股"
    
    # 清理旧日志（systemd 自动管理，无需手动清理）
    
    # 手动日志文件（如果使用 nohup 运行）
    tail -f scheduler.log

================================================================================
环境变量配置 (.env 文件)
================================================================================

# 已授权任务配置
ENABLE_DAILY_WORKFLOW=true     # 每日数据更新与选股任务
ENABLE_STREAMLIT=true          # Streamlit 可视化应用

# 日志配置
LOG_LEVEL=INFO                 # 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_FILE=scheduler.log         # 日志文件路径

================================================================================
任务授权说明
================================================================================

根据 dispatch.md 规则第 2 条（任务授权）：
- 不允许创建或调度任何未在配置中显式指定的任务
- 所有任务必须经过明确授权才能注册到调度器
- 当前已授权 2 个任务：
  1. 每日数据更新与选股任务
  2. Streamlit 可视化应用启动

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
    A: 检查 .env 中任务开关是否开启，确认当前是否在交易日
    
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

手动触发任务：
    # 在 Python 交互环境中
    from scheduler import daily_update_and_selection_task
    
    daily_update_and_selection_task()   # 手动执行每日数据更新与选股

================================================================================
"""
import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
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
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
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
        """
        # 提取 add_job 的参数（不属于 trigger 的参数）
        add_job_kwargs = {}
        if 'replace_existing' in kwargs:
            add_job_kwargs['replace_existing'] = kwargs.pop('replace_existing')
        if 'max_instances' in kwargs:
            add_job_kwargs['max_instances'] = kwargs.pop('max_instances')
        if 'misfire_grace_time' in kwargs:
            add_job_kwargs['misfire_grace_time'] = kwargs.pop('misfire_grace_time')
        
        if trigger == 'cron':
            trigger_obj = CronTrigger(**kwargs)
        elif trigger == 'date':
            trigger_obj = None
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
                max_instances=1,
                misfire_grace_time=60,
                **add_job_kwargs
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
        self.scheduler.shutdown(wait=wait)
        logger.info("调度器已关闭")


def run_subprocess(cmd: list, task_name: str, timeout: int = 3600) -> bool:
    """运行子进程任务
    
    Args:
        cmd: 命令列表
        task_name: 任务名称（用于日志）
        timeout: 超时时间（秒）
    
    Returns:
        bool: 是否成功
    """
    logger.info(f"[{task_name}] 执行命令：{' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        # 记录输出
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"[{task_name}] {line}")
        
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"[{task_name}] {line}")
        
        # 检查执行结果
        if result.returncode == 0:
            logger.info(f"✅ [{task_name}] 执行完成")
            return True
        else:
            logger.error(f"❌ [{task_name}] 执行失败，退出码：{result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"❌ [{task_name}] 执行超时（超过 {timeout} 秒）")
        raise
    except Exception as e:
        logger.error(f"❌ [{task_name}] 执行失败：{e}", exc_info=True)
        raise


def daily_update_and_selection_task():
    """每日数据更新与选股任务 - 已授权（周一至周五16:00执行）
    
    执行流程：
    1. 日线数据增量更新（每天）
    2. 周线数据增量更新（仅周五）
    3. 执行选股任务（每天）
    """
    today = datetime.now()
    is_friday = today.weekday() == 4  # 周五
    
    logger.info("=" * 60)
    logger.info(f"开始执行每日数据更新与选股任务 - {today.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"今天是：{['周一', '周二', '周三', '周四', '周五', '周六', '周日'][today.weekday()]}")
    logger.info("=" * 60)
    
    # ========== 步骤 1：日线数据增量更新（每天执行） ==========
    logger.info("【步骤 1/3】日线数据增量更新...")
    daily_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "app" / "build_dataset.py"),
        "--period", "d",
        "--update",
    ]
    
    try:
        run_subprocess(daily_cmd, "日线更新", timeout=3600)
    except Exception as e:
        logger.error(f"日线更新失败，停止后续步骤：{e}")
        raise
    
    # ========== 步骤 2：周线数据增量更新（仅周五执行） ==========
    if is_friday:
        logger.info("【步骤 2/3】周线数据增量更新（今天是周五）...")
        weekly_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "app" / "build_dataset.py"),
            "--period", "w",
            "--update",
        ]
        
        try:
            run_subprocess(weekly_cmd, "周线更新", timeout=3600)
        except Exception as e:
            logger.error(f"周线更新失败，继续执行选股任务：{e}")
            # 周线更新失败不阻断选股任务
    else:
        logger.info("【步骤 2/3】跳过周线更新（今天不是周五）")
    
    # ========== 步骤 3：执行选股任务（每天执行） ==========
    logger.info("【步骤 3/3】执行选股任务...")
    today_str = today.strftime("%Y-%m-%d")
    selection_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "selection" / "selection_ana.py"),
        "--date", today_str,
    ]
    
    try:
        run_subprocess(selection_cmd, "选股任务", timeout=7200)
    except Exception as e:
        logger.error(f"选股任务失败：{e}")
        raise
    
    logger.info("=" * 60)
    logger.info(f"✅ 每日数据更新与选股任务全部完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


def start_streamlit():
    """启动 Streamlit 可视化应用"""
    logger.info("=" * 60)
    logger.info("启动 Streamlit 可视化应用...")
    logger.info("=" * 60)
    
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "vis" / "financial_score_app.py"),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    
    try:
        # 使用 Popen 启动后台进程
        process = subprocess.Popen(
            streamlit_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        logger.info(f"✅ Streamlit 已启动 (PID: {process.pid})")
        logger.info(f"   访问地址: http://localhost:8501")
        logger.info(f"   命令: {' '.join(streamlit_cmd)}")
        
        return process
    
    except Exception as e:
        logger.error(f"❌ Streamlit 启动失败: {e}", exc_info=True)
        return None


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("统一任务调度器启动")
    logger.info("=" * 60)
    
    # 启动 Streamlit 可视化应用（后台运行）
    ENABLE_STREAMLIT = os.getenv('ENABLE_STREAMLIT', 'true').lower() == 'true'
    streamlit_process = None
    
    if ENABLE_STREAMLIT:
        streamlit_process = start_streamlit()
    else:
        logger.warning("Streamlit 可视化应用未启用（ENABLE_STREAMLIT=false）")
    
    # 创建调度器
    scheduler = TaskScheduler(use_async=False)
    
    # 从环境变量读取任务配置（仅支持已授权的任务）
    ENABLE_DAILY_WORKFLOW = os.getenv('ENABLE_DAILY_WORKFLOW', 'true').lower() == 'true'
    
    # 每日数据更新与选股任务 - 周一至周五 16:00 执行
    if ENABLE_DAILY_WORKFLOW:
        scheduler.register_task(
            name='每日数据更新与选股',
            func=daily_update_and_selection_task,
            trigger='cron',
            job_id='task_每日数据更新与选股',
            hour=16,
            minute=0,
            day_of_week='mon-fri',
        )
    else:
        logger.warning("每日数据更新与选股任务未启用（ENABLE_DAILY_WORKFLOW=false）")
    
    logger.info(f"已注册 {len(scheduler.tasks)} 个已授权任务")
    logger.info(f"任务列表：{list(scheduler.tasks.keys())}")
    
    try:
        # 启动调度器
        scheduler.start()
    finally:
        # 关闭时终止 Streamlit 进程
        if streamlit_process and streamlit_process.poll() is None:
            logger.info("正在终止 Streamlit 进程...")
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
            logger.info("Streamlit 进程已终止")


if __name__ == '__main__':
    main()
