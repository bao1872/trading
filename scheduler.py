#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一任务调度器

功能：集中管理所有定时调度任务
作者：项目团队
修改日期：2026-05-05
版本：v1.3.0

================================================================================
当前任务列表（共 4 类任务）
================================================================================

1. 盘前预览选股与预测 (task_盘前预览选股与预测) - [已授权]
   - 函数：pre_market_preview_task()
   - 触发器：Cron (每天 14:25)
   - 执行时间：交易日 14:25（收盘前，含节假日判断）
   - 交易日检查：通过 datasource.trade_calendar.is_trading_day() 判断，非交易日自动跳过
   - 执行流程：
        1. 日线K线增量更新（未收盘数据）
        2. Stop-Loss Clustering 选股
        3. SLC 模型预测（候选池 + 持仓股）
   - 配置：ENABLE_PREVIEW_WORKFLOW=true
   - 日志：scheduler.log 中搜索 "盘前预览"
   - 注意：14:25 数据为未收盘版本，15:10 正式任务会清理后重新计算
   - 状态：✅ 已授权并启用

2. 每日数据更新与策略任务 (task_每日数据更新与选股) - [已授权]
   - 函数：daily_update_and_selection_task()
   - 触发器：Cron (每天 15:10)
   - 执行时间：交易日 15:10（收盘后，含节假日判断）
   - 交易日检查：通过 datasource.trade_calendar.is_trading_day() 判断，非交易日自动跳过
   - 执行流程：
        0. 清理预览数据（14:25 产生的临时数据）
        1. 日线K线增量更新（每天）
        2. 周线K线增量更新（每天）
        3. 日线因子入库（已停用）
        4. 周线因子入库（已停用）
        5. DSA策略全流程（已停用）
        6. BSM选股 + BSM事件检测（已停用）
        7. Stop-Loss Clustering 选股
        8. SLC 模型预测（07_generate_daily_predictions，候选池 + 持仓股统一计算，写入 prediction_store + DB）
   - 配置：ENABLE_DAILY_WORKFLOW=true
   - 日志：scheduler.log 中搜索 "每日数据更新与选股"
   - 状态：✅ 已授权并启用

3. Streamlit 可视化应用启动 - [已授权]
   - 应用：vis/financial_score_app.py
   - 端口：8502
   - 访问地址：http://localhost:8502
   - 配置：ENABLE_STREAMLIT=true
   - 状态：✅ 已授权并启用（脚本启动时自动启动）

4. 盘中预测池监控任务 - [已授权]
   - 函数：pred_pool_monitor_task(freq)
   - 触发器：Cron (交易时段内按周期执行)
   - 执行时间：交易日 9:30-15:00（含节假日判断）
   - 交易日检查：通过 datasource.trade_calendar.is_trading_day() 判断，非交易日自动跳过
   - 检测周期：
     a. 15分钟：每15分钟整点执行（9:45, 10:00, 10:15... 14:45, 15:00）
     b. 60分钟：每天4个固定时间点（10:30, 11:30, 14:00, 14:55）
   - 检测对象：stop_loss_predictions 表上一交易日 Top 20（按 pred_sell_reg DESC）
   - 检测内容：MACD/Hist 常规/隐藏顶底背离、DSA VWAP触及
   - 推送方式：飞书消息
   - 配置：ENABLE_DIV_MONITOR=true
   - 日志：scheduler.log 中搜索 "预测池监控"
   - 状态：✅ 已授权并启用

5. 交易工作台 - [已停用]
   - 应用：vis/trading_workbench/app.py
   - 端口：8502
   - 配置：ENABLE_WORKBENCH=false
   - 状态：❌ 已停用（默认不启动，如需启用设 ENABLE_WORKBENCH=true）

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
ENABLE_PREVIEW_WORKFLOW=true   # 盘前预览选股与预测任务（14:25）
ENABLE_STREAMLIT=true          # Streamlit 可视化应用
ENABLE_WORKBENCH=false         # 交易工作台（已停用）
ENABLE_DIV_MONITOR=true        # 盘中预测池监控

# 日志配置
LOG_LEVEL=INFO                 # 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_FILE=scheduler.log         # 日志文件路径

================================================================================
任务授权说明
================================================================================

根据 dispatch.md 规则第 2 条（任务授权）：
- 不允许创建或调度任何未在配置中显式指定的任务
- 所有任务必须经过明确授权才能注册到调度器
- 当前已授权 4 个任务：
  1. 盘前预览选股与预测任务
  2. 每日数据更新与选股任务
  3. Streamlit 可视化应用启动
  4. 盘中预测池监控任务
- 已停用 1 个任务：
  5. 交易工作台（ENABLE_WORKBENCH=false）

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
    A: 检查 .env 中任务开关是否开启，确认当前是否为交易日（含节假日判断）
    
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
import functools
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from dotenv import load_dotenv

# 导入飞书配置
try:
    from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_USER_ID
except ImportError:
    FEISHU_APP_ID = None
    FEISHU_APP_SECRET = None
    FEISHU_USER_ID = None

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

from datasource.trade_calendar import is_trading_day


def trading_day_only(func):
    """装饰器：仅在交易日执行任务，非交易日跳过并记录日志"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_trading_day():
            logger.info(f"今天不是交易日，跳过任务：{func.__name__}")
            return
        return func(*args, **kwargs)
    return wrapper


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


def run_subprocess(cmd: list, task_name: str, timeout: int = 3600) -> None:
    """运行子进程任务

    Args:
        cmd: 命令列表
        task_name: 任务名称（用于日志）
        timeout: 超时时间（秒）

    Raises:
        RuntimeError: 子进程非零退出或超时
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

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"[{task_name}] {line}")

        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"[{task_name}] {line}")

        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else 'N/A'
            raise RuntimeError(
                f"[{task_name}] 执行失败，退出码: {result.returncode}\n"
                f"stderr: {stderr_tail}"
            )

        logger.info(f"✅ [{task_name}] 执行完成")

    except subprocess.TimeoutExpired:
        logger.error(f"❌ [{task_name}] 执行超时（超过 {timeout} 秒）")
        raise
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"❌ [{task_name}] 执行失败：{e}", exc_info=True)
        raise


def send_feishu_notification(title: str, content: str, is_error: bool = False):
    """发送飞书通知
    
    Args:
        title: 通知标题
        content: 通知内容
        is_error: 是否为错误通知
    """
    if not all([FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_USER_ID]):
        logger.warning("飞书配置不完整，跳过通知")
        return
    
    try:
        import requests
        import json
        
        # 获取 access_token
        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        token_payload = {
            "app_id": FEISHU_APP_ID,
            "app_secret": FEISHU_APP_SECRET
        }
        
        token_response = requests.post(token_url, json=token_payload, timeout=10)
        token_result = token_response.json()
        
        if token_result.get("code") != 0:
            logger.warning(f"获取飞书 access_token 失败：{token_result.get('msg')}")
            return
        
        access_token = token_result.get("tenant_access_token")
        
        # 发送消息
        emoji = "❌" if is_error else "✅"
        message = f"{emoji} {title}\n\n{content}"
        
        msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "receive_id": FEISHU_USER_ID,
            "msg_type": "text",
            "content": json.dumps({"text": message})
        }
        params = {"receive_id_type": "user_id"}
        
        msg_response = requests.post(msg_url, headers=headers, json=payload, params=params, timeout=10)
        msg_result = msg_response.json()
        
        if msg_result.get("code") == 0:
            logger.info(f"飞书通知发送成功：{title}")
        else:
            logger.warning(f"飞书通知发送失败：{msg_result.get('msg')}")
    
    except Exception as e:
        logger.warning(f"飞书通知发送失败：{e}")


def _cleanup_preview_data(target_date_str: str) -> dict:
    """清理预览任务产生的临时数据（14:25 未收盘数据）

    删除当天由盘前预览任务写入的临时数据，确保 15:10 正式任务重新拉取收盘数据。

    Args:
        target_date_str: 目标日期字符串，格式 YYYY-MM-DD

    Returns:
        dict: 各表删除行数 {'stock_k_data': N, 'stop_loss_selection': N, 'stop_loss_predictions': N, 'parquet_files': N}
    """
    from sqlalchemy import text
    from datasource.database import get_engine

    result = {}
    engine = get_engine()

    try:
        with engine.begin() as conn:
            kline_del = conn.execute(
                text("DELETE FROM stock_k_data WHERE freq = 'd' AND bar_time::date = :d"),
                {"d": target_date_str}
            )
            result['stock_k_data'] = kline_del.rowcount
            logger.info(f"[清理预览数据] stock_k_data 日线删除 {kline_del.rowcount} 条 (date={target_date_str})")

            sel_del = conn.execute(
                text("DELETE FROM stop_loss_selection WHERE selection_date = :d"),
                {"d": target_date_str}
            )
            result['stop_loss_selection'] = sel_del.rowcount
            logger.info(f"[清理预览数据] stop_loss_selection 删除 {sel_del.rowcount} 条 (date={target_date_str})")

            pred_del = conn.execute(
                text("DELETE FROM stop_loss_predictions WHERE prediction_date = :d"),
                {"d": target_date_str}
            )
            result['stop_loss_predictions'] = pred_del.rowcount
            logger.info(f"[清理预览数据] stop_loss_predictions 删除 {pred_del.rowcount} 条 (date={target_date_str})")
    finally:
        engine.dispose()

    import glob
    parquet_pattern = str(PROJECT_ROOT / "stop_experiment" / "output" / "prediction_store" / "**" / f"{target_date_str}.parquet")
    parquet_files = glob.glob(parquet_pattern, recursive=True)
    deleted_parquets = 0
    for f in parquet_files:
        try:
            os.remove(f)
            deleted_parquets += 1
            logger.info(f"[清理预览数据] 删除 parquet: {f}")
        except OSError as e:
            logger.warning(f"[清理预览数据] 删除 parquet 失败: {f}, {e}")
    result['parquet_files'] = deleted_parquets

    legacy_path = PROJECT_ROOT / "stop_experiment" / "output" / "predictions" / f"{target_date_str}.parquet"
    if legacy_path.exists():
        try:
            os.remove(str(legacy_path))
            result['parquet_files'] += 1
            logger.info(f"[清理预览数据] 删除 legacy parquet: {legacy_path}")
        except OSError as e:
            logger.warning(f"[清理预览数据] 删除 legacy parquet 失败: {e}")

    logger.info(f"[清理预览数据] 汇总: {result}")
    return result


@trading_day_only
def pre_market_preview_task():
    """盘前预览任务 - 已授权（交易日14:25执行）

    在收盘前拉取当天 K 线（未收盘数据），运行选股和预测，提前获得决策依据。
    15:10 正式任务会清理此任务产生的临时数据并重新计算。

    执行流程（3 步）：
    1. 日线K线增量更新（未收盘数据）
    2. Stop-Loss Clustering 选股
    3. SLC 模型预测（含持仓股）
    """
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    logger.info("=" * 60)
    logger.info(f"盘前预览任务开始 - {today_str} 14:25")
    logger.info("=" * 60)

    send_feishu_notification(
        f"盘前预览任务开始 - {today_str}",
        f"开始时间：{today.strftime('%H:%M:%S')}\n"
        f"注意：此为未收盘数据，仅供提前决策参考"
    )

    try:
        logger.info("[Step 1/3] 日线K线增量更新（未收盘）...")
        run_subprocess([
            sys.executable, str(PROJECT_ROOT / "app" / "build_dataset.py"),
            "--period", "d", "--update",
        ], "日线K线更新(预览)", timeout=3600)

        logger.info("[Step 2/3] Stop-Loss Clustering 选股（预览）...")
        try:
            run_subprocess([
                sys.executable, str(PROJECT_ROOT / "selection" / "selection_stop.py"),
            ], "SLC选股(预览)", timeout=3600)
            logger.info("✅ SLC 选股（预览）完成")
        except Exception as e:
            logger.error(f"SLC 选股（预览）失败: {e}")
            send_feishu_notification("SLC选股(预览)失败", str(e), is_error=True)

        logger.info("[Step 3/3] SLC 模型预测（预览）...")
        try:
            run_subprocess([
                sys.executable, "-m", "stop_experiment.pipeline.07_generate_daily_predictions",
                "--date", today_str, "--force",
            ], "SLC模型预测(预览)", timeout=3600)
            logger.info("✅ SLC 模型预测（预览）完成")
        except Exception as e:
            logger.error(f"SLC 模型预测（预览）失败: {e}")
            send_feishu_notification("SLC模型预测(预览)失败", str(e), is_error=True)

        end_time = datetime.now()
        duration = (end_time - today).total_seconds() / 60
        logger.info(f"✅ 盘前预览任务完成 ({duration:.1f}分钟)")
        send_feishu_notification(
            f"盘前预览任务完成 - {today_str}",
            f"完成时间：{end_time.strftime('%H:%M:%S')}\n"
            f"执行时长：{duration:.1f}分钟\n"
            f"⚠️ 数据为未收盘版本，15:10 将重新计算"
        )

    except Exception as e:
        logger.error(f"❌ 盘前预览任务失败: {e}")
        send_feishu_notification(
            f"盘前预览任务失败 - {today_str}",
            f"错误：{str(e)}",
            is_error=True
        )
        raise


@trading_day_only
def daily_update_and_selection_task():
    """每日数据更新与策略任务 - 已授权（交易日15:10执行）

    执行流程（7 步）：
    0. 清理预览数据（14:25 产生的临时数据）
    1. 日线K线增量更新
    2. 周线K线增量更新
    3. 日线因子入库 [已停用]
    4. 周线因子入库 [已停用]
    5. DSA策略全流程 [已停用]
    6. BSM选股 + BSM事件 [已停用]
    7. Stop-Loss Clustering 选股
    """
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    logger.info("=" * 60)
    logger.info(f"每日策略任务开始 - {today_str}")
    logger.info("=" * 60)

    send_feishu_notification(
        f"每日策略任务开始 - {today_str}",
        f"开始时间：{today.strftime('%H:%M:%S')}"
    )

    factors_ok = True
    weekly_factors_ok = True

    try:
        # ===== Step 0：清理预览数据 =====
        logger.info("[Step 0] 清理14:25预览任务产生的临时数据...")
        try:
            cleanup_result = _cleanup_preview_data(today_str)
            total_deleted = sum(v for v in cleanup_result.values())
            if total_deleted > 0:
                logger.info(f"✅ 清理预览数据完成: {cleanup_result}")
            else:
                logger.info("无预览数据需要清理（可能未执行预览任务）")
        except Exception as e:
            logger.warning(f"清理预览数据失败（非致命，继续执行）: {e}")

        # ===== Step 1/6：日线K线增量 =====
        logger.info("[Step 1/6] 日线K线增量更新...")
        run_subprocess([
            sys.executable, str(PROJECT_ROOT / "app" / "build_dataset.py"),
            "--period", "d", "--update",
        ], "日线K线更新", timeout=3600)

        # ===== Step 2/6：周线K线增量 =====
        logger.info("[Step 2/6] 周线K线增量更新...")
        try:
            run_subprocess([
                sys.executable, str(PROJECT_ROOT / "app" / "build_dataset.py"),
                "--period", "w", "--update",
            ], "周线K线更新", timeout=3600)
        except Exception as e:
            logger.warning(f"周线K线更新失败，继续: {e}")
            send_feishu_notification("周线K线更新失败", str(e), is_error=True)

        # ===== Step 3/6：日线因子入库 [已停用] =====
        logger.info("[Step 3/6] 日线因子入库 - 已停用，跳过")
        # 因子/事件不再每日写入 DB，实验请直接引用 factor_lib

        # ===== Step 4/6：周线因子入库 [已停用] =====
        logger.info("[Step 4/6] 周线因子入库 - 已停用，跳过")
        # 因子/事件不再每日写入 DB，实验请直接引用 factor_lib

        # ===== Step 5/6：DSA策略全流程 [已停用] =====
        logger.info("[Step 5/6] DSA策略全流程 - 已停用，跳过")
        # DSA策略已停用，如需启用请取消注释以下代码
        # if factors_ok and weekly_factors_ok:
        #     logger.info("[Step 5/6] DSA策略全流程...")
        #     try:
        #         run_subprocess([
        #             sys.executable, str(PROJECT_ROOT / "dsa_experiment" / "run_daily.py"),
        #             "--date", today_str, "--skip-factors", "--notify",
        #         ], "DSA策略全流程", timeout=7200)
        #     except Exception as e:
        #         logger.error(f"DSA策略全流程失败: {e}")
        #         send_feishu_notification("DSA策略失败", str(e), is_error=True)
        # else:
        #     logger.warning("因子计算不完整，跳过 DSA 策略全流程")

        # ===== Step 6/7：BSM选股 + BSM事件 [已停用] =====
        logger.info("[Step 6/7] BSM选股 + BSM事件检测 - 已停用，跳过")
        # BSM选股已停用，如需启用请取消注释以下代码
        # logger.info("[Step 6/7] BSM选股...")
        # try:
        #     run_subprocess([
        #         sys.executable, str(PROJECT_ROOT / "selection" / "selection_ana.py"),
        #     ], "BSM选股", timeout=3600)
        # except Exception as e:
        #     logger.error(f"BSM选股失败: {e}")
        #     send_feishu_notification("BSM选股失败", str(e), is_error=True)
        #
        # logger.info("[Step 6/7] BSM事件检测...")
        # try:
        #     from selection.watchlist_event_detection import detect_and_save_bsm_events
        #     detect_and_save_bsm_events()
        #     logger.info("✅ BSM事件检测完成")
        # except Exception as e:
        #     logger.error(f"BSM事件检测失败: {e}")
        #     send_feishu_notification("BSM事件检测失败", str(e), is_error=True)

        # ===== Step 7/8：Stop-Loss Clustering 选股 =====
        logger.info("[Step 7/8] Stop-Loss Clustering 选股...")
        try:
            run_subprocess([
                sys.executable, str(PROJECT_ROOT / "selection" / "selection_stop.py"),
            ], "SLC选股", timeout=3600)
            logger.info("✅ Stop-Loss Clustering 选股完成")
        except Exception as e:
            logger.error(f"Stop-Loss Clustering 选股失败: {e}")
            send_feishu_notification("SLC选股失败", str(e), is_error=True)

        # ===== Step 8/9：SLC 模型预测 =====
        logger.info("[Step 8/9] SLC 模型预测...")
        try:
            run_subprocess([
                sys.executable, "-m", "stop_experiment.pipeline.07_generate_daily_predictions",
                "--date", today_str, "--force",
            ], "SLC模型预测", timeout=3600)
            logger.info("✅ SLC 模型预测完成")
        except Exception as e:
            logger.error(f"SLC 模型预测失败: {e}")
            send_feishu_notification("SLC模型预测失败", str(e), is_error=True)

        # 完成
        end_time = datetime.now()
        duration = (end_time - today).total_seconds() / 60
        logger.info(f"✅ 每日策略任务完成 ({duration:.1f}分钟)")
        send_feishu_notification(
            f"每日策略任务完成 - {today_str}",
            f"完成时间：{end_time.strftime('%H:%M:%S')}\n"
            f"执行时长：{duration:.1f}分钟\n"
            f"因子状态: 1d={'✅' if factors_ok else '❌'} 1w={'✅' if weekly_factors_ok else '❌'}"
        )

    except Exception as e:
        logger.error(f"❌ 每日策略任务失败: {e}")
        send_feishu_notification(
            f"每日策略任务失败 - {today_str}",
            f"错误：{str(e)}",
            is_error=True
        )
        raise


@trading_day_only
def pred_pool_monitor_task(freq: str):
    """盘中预测池监控任务

    Args:
        freq: 检测周期（5m/15m/60m）
    """
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')

    logger.info(f"=" * 60)
    logger.info(f"开始执行预测池监控任务 (周期: {freq}) - {now_str}")
    logger.info(f"=" * 60)

    try:
        from app.monitoring import run_monitor
        run_monitor(freq)

        logger.info(f"✅ 预测池监控任务完成 (周期: {freq})")

    except Exception as e:
        logger.error(f"❌ 预测池监控任务失败 (周期: {freq})：{e}", exc_info=True)
        send_feishu_notification(
            f"预测池监控失败 - {freq}",
            f"时间：{now.strftime('%H:%M:%S')}\n错误：{str(e)}",
            is_error=True
        )


def start_streamlit():
    logger.info("=" * 60)
    logger.info("启动 Streamlit 可视化应用...")
    logger.info("=" * 60)
    
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "vis" / "financial_score_app.py"),
        "--server.port", "8502",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    
    try:
        process = subprocess.Popen(
            streamlit_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        logger.info(f"✅ Streamlit 已启动 (PID: {process.pid})")
        logger.info(f"   访问地址: http://localhost:8502")
        logger.info(f"   命令: {' '.join(streamlit_cmd)}")
        
        return process
    
    except Exception as e:
        logger.error(f"❌ Streamlit 启动失败: {e}", exc_info=True)
        return None


def start_trading_workbench():
    logger.info("=" * 60)
    logger.info("启动交易工作台 Streamlit 应用...")
    logger.info("=" * 60)
    
    workbench_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "vis" / "trading_workbench" / "app.py"),
        "--server.port", "8502",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    
    try:
        process = subprocess.Popen(
            workbench_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        logger.info(f"✅ 交易工作台已启动 (PID: {process.pid})")
        logger.info(f"   访问地址: http://localhost:8502")
        logger.info(f"   命令: {' '.join(workbench_cmd)}")
        
        return process
    
    except Exception as e:
        logger.error(f"❌ 交易工作台启动失败: {e}", exc_info=True)
        return None


def main():
    """主函数"""
    from datetime import datetime
    
    logger.info("=" * 60)
    logger.info("统一任务调度器启动")
    logger.info("=" * 60)
    
    # 发送服务启动通知
    start_time = datetime.now()
    
    # 启动 Streamlit 可视化应用（后台运行）
    ENABLE_STREAMLIT = os.getenv('ENABLE_STREAMLIT', 'true').lower() == 'true'
    streamlit_process = None
    streamlit_status = "未启用"
    
    if ENABLE_STREAMLIT:
        streamlit_process = start_streamlit()
        if streamlit_process:
            streamlit_status = f"已启动 (PID: {streamlit_process.pid})"
        else:
            streamlit_status = "启动失败"
    else:
        logger.warning("Streamlit 可视化应用未启用（ENABLE_STREAMLIT=false）")

    # 启动交易工作台（后台运行，端口 8502）
    ENABLE_WORKBENCH = os.getenv('ENABLE_WORKBENCH', 'false').lower() == 'true'
    workbench_process = None
    workbench_status = "未启用"

    if ENABLE_WORKBENCH:
        workbench_process = start_trading_workbench()
        if workbench_process:
            workbench_status = f"已启动 (PID: {workbench_process.pid})"
        else:
            workbench_status = "启动失败"
    else:
        logger.warning("交易工作台未启用（ENABLE_WORKBENCH=false）")
    
    # 创建调度器
    scheduler = TaskScheduler(use_async=False)
    
    # 从环境变量读取任务配置（仅支持已授权的任务）
    ENABLE_DAILY_WORKFLOW = os.getenv('ENABLE_DAILY_WORKFLOW', 'true').lower() == 'true'
    ENABLE_PREVIEW_WORKFLOW = os.getenv('ENABLE_PREVIEW_WORKFLOW', 'true').lower() == 'true'
    ENABLE_DIV_MONITOR = os.getenv('ENABLE_DIV_MONITOR', 'true').lower() == 'true'

    # 盘前预览任务 - 周一至周五 14:25 执行
    if ENABLE_PREVIEW_WORKFLOW:
        scheduler.register_task(
            name='盘前预览选股与预测',
            func=pre_market_preview_task,
            trigger='cron',
            job_id='task_盘前预览选股与预测',
            hour=14,
            minute=25,
            day_of_week='mon-fri',
        )
    else:
        logger.warning("盘前预览任务未启用（ENABLE_PREVIEW_WORKFLOW=false）")

    # 每日数据更新与选股任务 - 周一至周五 15:10 执行
    if ENABLE_DAILY_WORKFLOW:
        scheduler.register_task(
            name='每日数据更新与选股',
            func=daily_update_and_selection_task,
            trigger='cron',
            job_id='task_每日数据更新与选股',
            hour=15,
            minute=10,
            day_of_week='mon-fri',
        )
    else:
        logger.warning("每日数据更新与选股任务未启用（ENABLE_DAILY_WORKFLOW=false）")
    
    # 盘中预测池监控任务
    if ENABLE_DIV_MONITOR:
        # 15分钟周期：每15分钟整点执行（9:45-14:55）
        for h in range(9, 15):
            for m in [0, 15, 30, 45]:
                if h == 9 and m < 45:
                    continue
                if h == 11 and m > 30:
                    continue
                if h == 12:
                    continue
                if h == 13 and m < 15:
                    continue
                scheduler.register_task(
                    name=f'预测池监控15m_{h:02d}:{m:02d}',
                    func=lambda h=h, m=m: pred_pool_monitor_task('15m'),
                    trigger='cron',
                    job_id=f'task_pred_pool_15m_{h:02d}{m:02d}',
                    hour=h,
                    minute=m,
                    day_of_week='mon-fri',
                )

        # 15分钟周期：收盘前5分钟额外执行
        scheduler.register_task(
            name='预测池监控15m_14:55',
            func=lambda: pred_pool_monitor_task('15m'),
            trigger='cron',
            job_id='task_pred_pool_15m_1455',
            hour=14,
            minute=55,
            day_of_week='mon-fri',
        )

        # 60分钟周期：每天4个固定时间点（10:30, 11:30, 14:00, 14:55）
        for h, m in [(10, 30), (11, 30), (14, 0), (14, 55)]:
            scheduler.register_task(
                name=f'预测池监控60m_{h:02d}:{m:02d}',
                func=lambda h=h, m=m: pred_pool_monitor_task('60m'),
                trigger='cron',
                job_id=f'task_pred_pool_60m_{h:02d}{m:02d}',
                hour=h,
                minute=m,
                day_of_week='mon-fri',
            )
    else:
        logger.warning("预测池监控任务未启用（ENABLE_DIV_MONITOR=false）")
    
    logger.info(f"已注册 {len(scheduler.tasks)} 个已授权任务")
    logger.info(f"任务列表：{list(scheduler.tasks.keys())}")
    
    # 发送服务启动成功通知
    task_list = ", ".join(scheduler.tasks.keys()) if scheduler.tasks else "无"
    send_feishu_notification(
        "📊 选股调度服务已启动",
        f"启动时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Streamlit状态：{streamlit_status}\n"
        f"已注册任务：{task_list}\n"
        f"每日选股时间：15:10（周一至周五）",
        is_error=False
    )
    
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
        # 关闭时终止交易工作台进程
        if workbench_process and workbench_process.poll() is None:
            logger.info("正在终止交易工作台进程...")
            workbench_process.terminate()
            workbench_process.wait(timeout=5)
            logger.info("交易工作台进程已终止")


if __name__ == '__main__':
    main()
