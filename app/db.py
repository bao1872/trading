# -*- coding: utf-8 -*-
"""
数据库对接模块 - 标准化的数据库操作接口

所有数据库操作统一引用此模块，提供：
- 连接池管理
- 会话管理（上下文管理器）
- 批量写入（upsert）
- 批量查询
- 表清空
- 事务支持

使用示例:
    from app.db import get_session, bulk_upsert, truncate_table, query_df
    
    with get_session() as session:
        bulk_upsert(session, Model, df, unique_keys=["ts_code", "freq"])
"""
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    """获取数据库引擎（单例模式）"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=False,
        )
    return _engine


def get_session_factory():
    """获取会话工厂（单例模式）"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    return _SessionLocal


@contextmanager
def get_session() -> Session:
    """
    获取数据库会话（上下文管理器）
    
    使用示例:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库操作异常: {e}")
        raise
    finally:
        session.close()


def execute_sql(sql: str, params: Optional[Dict] = None) -> None:
    """
    执行原生SQL语句
    
    Args:
        sql: SQL语句
        params: 参数字典
    """
    with get_session() as session:
        session.execute(text(sql), params or {})


def truncate_table(session: Session, model: Type) -> None:
    """
    清空表数据（TRUNCATE）
    
    Args:
        session: 数据库会话
        model: SQLAlchemy模型类
    """
    table_name = model.__tablename__
    session.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE"))
    logger.info(f"已清空表: {table_name}")


def delete_by_filter(session: Session, model: Type, filters: Dict[str, Any]) -> int:
    """
    按条件删除数据
    
    Args:
        session: 数据库会话
        model: SQLAlchemy模型类
        filters: 过滤条件
        
    Returns:
        删除的行数
    """
    query = session.query(model)
    for key, value in filters.items():
        if hasattr(model, key):
            query = query.filter(getattr(model, key) == value)
    
    count = query.count()
    query.delete(synchronize_session=False)
    logger.info(f"删除 {model.__tablename__} 表 {count} 条记录, 条件: {filters}")
    return count


def bulk_insert(session: Session, model: Type, df: pd.DataFrame) -> int:
    """
    批量插入数据
    
    Args:
        session: 数据库会话
        model: SQLAlchemy模型类
        df: 数据DataFrame
        
    Returns:
        插入的行数
    """
    if df.empty:
        return 0
    
    records = df.to_dict(orient="records")
    session.bulk_insert_mappings(model, records)
    logger.info(f"批量插入 {model.__tablename__} 表 {len(records)} 条记录")
    return len(records)


def bulk_upsert(
    session: Session, 
    model: Type, 
    df: pd.DataFrame, 
    unique_keys: List[str],
    batch_size: int = 1000
) -> int:
    """
    批量写入数据（upsert模式：存在则更新，不存在则插入）
    
    Args:
        session: 数据库会话
        model: SQLAlchemy模型类
        df: 数据DataFrame
        unique_keys: 唯一键列表
        batch_size: 批量处理大小
        
    Returns:
        写入的行数
    """
    if df.empty:
        return 0
    
    table_name = model.__tablename__
    columns = list(df.columns)
    
    non_key_columns = [c for c in columns if c not in unique_keys]
    
    if not non_key_columns:
        return bulk_insert(session, model, df)
    
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key_columns])
    key_clause = ", ".join(unique_keys)
    col_clause = ", ".join(columns)
    placeholders = ", ".join([f":{c}" for c in columns])
    
    sql = f"""
        INSERT INTO {table_name} ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT ({key_clause}) DO UPDATE SET {update_clause}
    """
    
    records = df.to_dict(orient="records")
    total = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        session.execute(text(sql), batch)
        total += len(batch)
    
    logger.info(f"批量写入 {table_name} 表 {total} 条记录 (upsert)")
    return total


def query_df(
    session: Session,
    model: Type,
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> pd.DataFrame:
    """
    查询数据返回DataFrame
    
    Args:
        session: 数据库会话
        model: SQLAlchemy模型类
        filters: 过滤条件
        order_by: 排序字段
        limit: 返回行数限制
        offset: 偏移量
        
    Returns:
        查询结果DataFrame
    """
    query = session.query(model)
    
    if filters:
        for key, value in filters.items():
            if hasattr(model, key):
                if isinstance(value, (list, tuple)):
                    query = query.filter(getattr(model, key).in_(value))
                else:
                    query = query.filter(getattr(model, key) == value)
    
    if order_by:
        desc = order_by.startswith("-")
        col_name = order_by.lstrip("-")
        if hasattr(model, col_name):
            col = getattr(model, col_name)
            query = query.order_by(col.desc() if desc else col.asc())
    
    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)
    
    result = query.all()
    
    if not result:
        columns = [c.name for c in model.__table__.columns]
        return pd.DataFrame(columns=columns)
    
    records = [{c.name: getattr(row, c.name) for c in model.__table__.columns} for row in result]
    return pd.DataFrame(records)


def query_sql(
    session: Session,
    sql: str,
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    执行SQL查询返回DataFrame
    
    Args:
        session: 数据库会话
        sql: SQL查询语句
        params: 参数字典
        
    Returns:
        查询结果DataFrame
    """
    result = session.execute(text(sql), params or {})
    columns = result.keys()
    rows = result.fetchall()
    
    if not rows:
        return pd.DataFrame(columns=columns)
    
    return pd.DataFrame(rows, columns=columns)


def count_records(
    session: Session,
    model: Type,
    filters: Optional[Dict[str, Any]] = None
) -> int:
    """
    统计记录数
    
    Args:
        session: 数据库会话
        model: SQLAlchemy模型类
        filters: 过滤条件
        
    Returns:
        记录数
    """
    query = session.query(model)
    
    if filters:
        for key, value in filters.items():
            if hasattr(model, key):
                query = query.filter(getattr(model, key) == value)
    
    return query.count()


def table_exists(session: Session, table_name: str) -> bool:
    """
    检查表是否存在
    
    Args:
        session: 数据库会话
        table_name: 表名
        
    Returns:
        是否存在
    """
    sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = :table_name
        )
    """
    result = session.execute(text(sql), {"table_name": table_name})
    return result.scalar()


def get_table_columns(session: Session, table_name: str) -> List[str]:
    """
    获取表的列名列表
    
    Args:
        session: 数据库会话
        table_name: 表名
        
    Returns:
        列名列表
    """
    sql = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = :table_name
        ORDER BY ordinal_position
    """
    result = session.execute(text(sql), {"table_name": table_name})
    return [row[0] for row in result.fetchall()]


if __name__ == "__main__":
    with get_session() as session:
        result = session.execute(text("SELECT 1 as test"))
        print(f"数据库连接测试: {result.fetchone()}")
        
        print(f"表 stock_div_features 存在: {table_exists(session, 'stock_div_features')}")
        print(f"表 stock_amp_features 存在: {table_exists(session, 'stock_amp_features')}")
