# -*- coding: utf-8 -*-
"""
统一日志模块 - 实现标准化日志输出与管理

所有工程化模块必须使用此日志模块输出日志，确保日志格式统一、便于排查问题。

使用示例:
    from app.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("开始处理数据")
    logger.warning("数据不完整")
    logger.error("处理失败", exc_info=True)
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


_logger_cache: dict = {}


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        level: 日志级别，默认为 INFO
        
    Returns:
        配置好的日志记录器
    """
    if name in _logger_cache:
        return _logger_cache[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level or logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(
            LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    _logger_cache[name] = logger
    return logger


def set_log_level(level: int):
    """
    设置全局日志级别
    
    Args:
        level: 日志级别 (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
    """
    for logger in _logger_cache.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)


root_logger = get_logger("src")
