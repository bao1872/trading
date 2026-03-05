# -*- coding: utf-8 -*-
"""
数据库迁移脚本：为 stock_amp_features 表添加 name 字段

使用方法:
  python -m app.migrations.add_name_to_amp_features
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db import get_session
from sqlalchemy import text
from app.logger import get_logger

logger = get_logger(__name__)


def add_name_column():
    """为 stock_amp_features 表添加 name 字段"""
    with get_session() as session:
        # 检查字段是否已存在
        check_sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'stock_amp_features' AND column_name = 'name'
            )
        """
        result = session.execute(text(check_sql))
        exists = result.scalar()
        
        if exists:
            logger.info("字段 name 已存在，无需添加")
            return
        
        # 添加 name 字段
        alter_sql = """
            ALTER TABLE stock_amp_features 
            ADD COLUMN name VARCHAR(50)
        """
        session.execute(text(alter_sql))
        logger.info("成功添加 name 字段到 stock_amp_features 表")
        
        # 为现有数据填充 name（从股票池读取）
        logger.info("开始为现有数据填充 name 字段...")
        
        # 读取股票池
        stock_pool_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'stock_concepts_cache.xlsx'
        )
        
        import pandas as pd
        df_pool = pd.read_excel(stock_pool_path)
        stock_dict = dict(zip(df_pool['ts_code'], df_pool['name']))
        
        # 批量更新
        updated_count = 0
        for ts_code, name in stock_dict.items():
            update_sql = """
                UPDATE stock_amp_features 
                SET name = :name 
                WHERE ts_code = :ts_code AND name IS NULL
            """
            result = session.execute(text(update_sql), {"ts_code": ts_code, "name": name})
            updated_count += result.rowcount or 0
        
        logger.info(f"已更新 {updated_count} 条记录的 name 字段")


if __name__ == "__main__":
    logger.info("开始执行数据库迁移...")
    add_name_column()
    logger.info("数据库迁移完成")
