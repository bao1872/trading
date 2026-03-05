#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票列表管理模块

提供从数据库读取股票列表的功能，替代直接读取 stock.xlsx 文件
"""

import os
import sys
from typing import List, Dict

# 添加项目根目录到路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from app.db import get_session
from sqlalchemy import text


def get_stock_list_from_db() -> List[str]:
    """
    从数据库获取股票名称列表
    
    Returns:
        股票名称列表
    """
    with get_session() as session:
        result = session.execute(text("SELECT stock_name FROM stock_list ORDER BY id"))
        return [row.stock_name for row in result.fetchall()]


def get_stock_list_with_codes() -> dict:
    """
    从数据库获取股票名称和代码的映射
    
    Returns:
        {股票名称：股票代码} 的字典
    """
    with get_session() as session:
        result = session.execute(text("SELECT stock_name, ts_code FROM stock_list ORDER BY id"))
        return {row.stock_name: row.ts_code for row in result.fetchall()}


def get_stock_cache_from_db() -> Dict[str, str]:
    """
    从数据库获取股票概念缓存（名称到代码的映射）
    
    Returns:
        {股票名称：股票代码} 的字典
    """
    with get_session() as session:
        # 尝试从 stock_list 表读取
        result = session.execute(text("""
            SELECT stock_name, ts_code 
            FROM stock_list 
            WHERE ts_code IS NOT NULL 
            ORDER BY id
        """))
        cache = {row.stock_name: row.ts_code for row in result.fetchall()}
        
        # 如果 stock_list 中没有 ts_code，尝试从 stock_concepts_cache 读取
        if not cache:
            try:
                result = session.execute(text("""
                    SELECT name, ts_code 
                    FROM stock_concepts_cache 
                    ORDER BY id
                """))
                cache = {row.name: row.ts_code for row in result.fetchall()}
            except Exception:
                # stock_concepts_cache 表不存在或为空
                pass
        
        return cache


def get_stock_count() -> int:
    """
    获取数据库中股票数量
    
    Returns:
        股票数量
    """
    with get_session() as session:
        result = session.execute(text("SELECT COUNT(*) FROM stock_list"))
        return result.scalar()


if __name__ == "__main__":
    # 测试
    print("📊 股票列表管理模块测试")
    print(f"📋 股票数量：{get_stock_count()}")
    print(f"📋 股票列表：{get_stock_list_from_db()}")
