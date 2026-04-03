# -*- coding: utf-8 -*-
"""
数据库操作模块
提供统一的数据库连接、会话管理和批量操作功能
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session

# 从配置导入数据库URL
from config import DATABASE_URL

# 配置日志
logger = logging.getLogger(__name__)

# 全局引擎实例（延迟初始化）
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """获取数据库引擎（单例模式）"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,  # 连接前ping测试，避免使用已断开的连接
            pool_recycle=3600,   # 连接1小时后回收
            echo=False
        )
    return _engine


# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器
    
    使用示例:
        with get_session() as session:
            session.execute(...)
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库操作失败: {e}")
        raise
    finally:
        session.close()


def bulk_insert(
    conn: Any,
    table_name: str,
    df: pd.DataFrame,
    auto_commit: bool = True
) -> int:
    """
    批量插入数据

    Args:
        conn: SQLAlchemy Connection 对象
        table_name: 表名
        df: 数据DataFrame
        auto_commit: 是否自动提交事务（默认True）

    Returns:
        插入的行数
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    columns = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in columns])
    col_clause = ", ".join(columns)

    sql = f"INSERT INTO {table_name} ({col_clause}) VALUES ({placeholders})"

    try:
        for i in range(0, len(records), 1000):
            batch = records[i:i + 1000]
            conn.execute(text(sql), batch)

        if auto_commit:
            conn.commit()
        logger.info(f"批量插入 {table_name} 表 {len(records)} 条记录")
        return len(records)
    except Exception as e:
        if auto_commit:
            conn.rollback()
        logger.warning(f"批量插入失败: {e}")
        raise


def bulk_upsert(
    conn: Any,
    table_name: str,
    df: pd.DataFrame,
    unique_keys: List[str],
    auto_commit: bool = True
) -> int:
    """
    批量写入数据（upsert模式：存在则更新，不存在则插入）

    Args:
        conn: SQLAlchemy Connection 对象
        table_name: 表名
        df: 数据DataFrame
        unique_keys: 唯一键列表
        auto_commit: 是否自动提交事务（默认True）

    Returns:
        写入的行数
    """
    if df.empty:
        return 0

    columns = list(df.columns)
    non_key_columns = [c for c in columns if c not in unique_keys]

    if not non_key_columns:
        return bulk_insert(conn, table_name, df, auto_commit=auto_commit)

    return _bulk_upsert_sqlalchemy(conn, table_name, df, unique_keys, auto_commit=auto_commit)


def _bulk_upsert_sqlalchemy(
    conn: Any,
    table_name: str,
    df: pd.DataFrame,
    unique_keys: List[str],
    auto_commit: bool = True
) -> int:
    """使用 SQLAlchemy 原生方式批量upsert"""
    columns = list(df.columns)
    non_key_columns = [c for c in columns if c not in unique_keys]

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

    try:
        for i in range(0, len(records), 1000):
            batch = records[i:i + 1000]
            conn.execute(text(sql), batch)
            total += len(batch)

        if auto_commit:
            conn.commit()
        logger.info(f"批量写入 {table_name} 表 {total} 条记录 (upsert)")
        return total
    except Exception as e:
        if auto_commit:
            conn.rollback()
        logger.warning(f"批量写入 {table_name} 失败: {e}")
        raise


def query_df(
    conn: Any,
    table_name: str,
    columns: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> pd.DataFrame:
    """
    查询数据返回DataFrame

    Args:
        conn: SQLAlchemy Connection 对象
        table_name: 表名
        columns: 要查询的列名列表，None 表示所有列
        filters: 过滤条件，支持比较操作符 (>=, <=, >, <, !=)
        order_by: 排序字段（加 - 表示降序）
        limit: 返回行数限制
        offset: 偏移量

    Returns:
        查询结果DataFrame
    """
    import re

    # 构建列名
    if columns:
        col_clause = ", ".join(columns)
    else:
        col_clause = "*"

    # 构建WHERE子句
    where_clauses = []
    params = {}

    if filters:
        for key, value in filters.items():
            # 支持操作符: >=, <=, >, <, !=, =
            match = re.match(r'^(\w+)\s*(>=|<=|>|<|!=|=)?$', key)
            if match:
                col_name = match.group(1)
                operator = match.group(2) or '='
                safe_key = col_name.replace(' ', '_').replace('-', '_')
                placeholder = f":{safe_key}"
                where_clauses.append(f"{col_name} {operator} {placeholder}")
                params[safe_key] = value

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    # 构建ORDER BY
    order_sql = ""
    if order_by:
        # 支持多字段排序，格式: "+field1,-field2" 或 "field1,-field2"
        order_parts = []
        for field in order_by.split(","):
            field = field.strip()
            if field.startswith("-"):
                order_parts.append(f"{field[1:]} DESC")
            elif field.startswith("+"):
                order_parts.append(f"{field[1:]} ASC")
            else:
                order_parts.append(f"{field} ASC")
        order_sql = "ORDER BY " + ", ".join(order_parts)

    # 构建LIMIT和OFFSET
    limit_sql = f"LIMIT {limit}" if limit else ""
    offset_sql = f"OFFSET {offset}" if offset else ""

    # 组装SQL
    sql = f"SELECT {col_clause} FROM {table_name} {where_sql} {order_sql} {limit_sql} {offset_sql}".strip()

    # 执行查询
    result = conn.execute(text(sql), params)
    rows = result.fetchall()

    # 转换为DataFrame
    if rows:
        df = pd.DataFrame(rows, columns=result.keys())
    else:
        df = pd.DataFrame(columns=result.keys() if result.keys() else [])

    return df


def table_exists(conn: Any, table_name: str) -> bool:
    """检查表是否存在"""
    if DATABASE_URL.startswith("postgresql"):
        sql = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"
    else:
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"

    result = conn.execute(text(sql), {"table_name": table_name})
    return bool(result.scalar())


def get_table_columns(conn: Any, table_name: str) -> List[str]:
    """获取表的列名列表"""
    if DATABASE_URL.startswith("postgresql"):
        sql = "SELECT column_name FROM information_schema.columns WHERE table_name = :table_name"
        result = conn.execute(text(sql), {"table_name": table_name})
        return [row[0] for row in result.fetchall()]
    else:
        # SQLite
        sql = f"PRAGMA table_info({table_name})"
        result = conn.execute(text(sql))
        return [row[1] for row in result.fetchall()]


# 为了保持向后兼容，保留旧的函数签名
def get_session_legacy() -> Generator[Session, None, None]:
    """兼容旧代码的会话获取方式"""
    return get_session()


def execute_sql(sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    执行SQL语句

    Args:
        sql: SQL语句
        params: SQL参数

    Returns:
        执行结果
    """
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
        return result
