# -*- coding: utf-8 -*-
"""
数据库操作模块 - 标准化的数据库操作接口

所有数据库操作统一引用此模块，提供：
- 连接池管理
- 会话管理（上下文管理器）
- 批量写入（upsert）
- 批量查询
- 表清空
- 事务支持

使用示例:
    from datasource.database import get_session, bulk_upsert, truncate_table, query_df

    with get_session() as session:
        bulk_upsert(session, Model, df, unique_keys=["ts_code", "freq"])
"""
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """获取数据库引擎（单例模式）"""
    global _engine
    if _engine is None:
        if DATABASE_URL.startswith("sqlite"):
            _engine = create_engine(
                DATABASE_URL,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
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


def get_session_factory() -> sessionmaker:
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


def truncate_table(session: Session, table_name: str) -> None:
    """
    清空表数据（TRUNCATE）

    Args:
        session: 数据库会话
        table_name: 表名
    """
    if DATABASE_URL.startswith("sqlite"):
        session.execute(text(f"DELETE FROM {table_name}"))
    else:
        session.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE"))
    logger.info(f"已清空表: {table_name}")


def delete_by_filter(session: Session, table_name: str, filters: Dict[str, Any]) -> int:
    """
    按条件删除数据

    Args:
        session: 数据库会话
        table_name: 表名
        filters: 过滤条件

    Returns:
        删除的行数
    """
    where_clauses = [f"{key} = :{key}" for key in filters.keys()]
    where_sql = " AND ".join(where_clauses)
    sql = f"DELETE FROM {table_name} WHERE {where_sql}"

    result = session.execute(text(sql), filters)
    logger.info(f"删除 {table_name} 表 {result.rowcount} 条记录, 条件: {filters}")
    return result.rowcount


def bulk_insert(session: Session, table_name: str, df: pd.DataFrame) -> int:
    """
    批量插入数据

    Args:
        session: 数据库会话
        table_name: 表名
        df: 数据DataFrame

    Returns:
        插入的行数
    """
    if df.empty:
        return 0

    if not DATABASE_URL.startswith("postgresql"):
        return _bulk_insert_sqlalchemy(session, table_name, df)

    return _bulk_insert_psycopg2(session, table_name, df)


def _bulk_insert_sqlalchemy(session: Session, table_name: str, df: pd.DataFrame) -> int:
    """使用 SQLAlchemy 原生方式批量插入"""
    records = df.to_dict(orient="records")
    columns = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in columns])
    col_clause = ", ".join(columns)

    sql = f"INSERT INTO {table_name} ({col_clause}) VALUES ({placeholders})"

    for i in range(0, len(records), 1000):
        batch = records[i:i + 1000]
        session.execute(text(sql), batch)

    logger.info(f"批量插入 {table_name} 表 {len(records)} 条记录")
    return len(records)


def _bulk_insert_psycopg2(session: Session, table_name: str, df: pd.DataFrame) -> int:
    """使用 psycopg2 原生批量插入"""
    from psycopg2.extras import execute_batch

    if df.empty:
        return 0

    columns = list(df.columns)
    col_clause = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table_name} ({col_clause}) VALUES ({placeholders})"

    records = [tuple(row) for row in df.values]

    try:
        conn = session.connection()
        raw_conn = conn.connection.driver_connection
        cursor = raw_conn.cursor()
        execute_batch(cursor, sql, records, batch_size=5000)
        session.commit()
        logger.info(f"批量插入 {table_name} 表 {len(records)} 条记录")
        return len(records)
    except Exception as e:
        session.rollback()
        logger.warning(f"psycopg2 批量插入失败，回退到 SQLAlchemy 方式: {e}")
        return _bulk_insert_sqlalchemy(session, table_name, df)


def bulk_upsert(
    session: Session,
    table_name: str,
    df: pd.DataFrame,
    unique_keys: List[str],
    batch_size: int = 1000
) -> int:
    """
    批量写入数据（upsert模式：存在则更新，不存在则插入）

    Args:
        session: 数据库会话
        table_name: 表名
        df: 数据DataFrame
        unique_keys: 唯一键列表
        batch_size: 批量处理大小

    Returns:
        写入的行数
    """
    if df.empty:
        return 0

    columns = list(df.columns)

    non_key_columns = [c for c in columns if c not in unique_keys]

    if not non_key_columns:
        return bulk_insert(session, table_name, df)

    if not DATABASE_URL.startswith("postgresql"):
        return _bulk_upsert_sqlalchemy(session, table_name, df, unique_keys)

    return _bulk_upsert_psycopg2(session, table_name, df, unique_keys, batch_size)


def _bulk_upsert_sqlalchemy(
    session: Session,
    table_name: str,
    df: pd.DataFrame,
    unique_keys: List[str]
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

    for i in range(0, len(records), 1000):
        batch = records[i:i + 1000]
        try:
            session.execute(text(sql), batch)
            total += len(batch)
        except Exception as e:
            session.rollback()
            logger.warning(f"批量写入 {table_name} 第 {i//1000 + 1} 批失败，已回滚: {e}")

    if total > 0:
        session.commit()
    logger.info(f"批量写入 {table_name} 表 {total} 条记录 (upsert)")
    return total


def _bulk_upsert_psycopg2(
    session: Session,
    table_name: str,
    df: pd.DataFrame,
    unique_keys: List[str],
    batch_size: int = 1000
) -> int:
    """使用 psycopg2 execute_batch + 临时表实现高效 upsert"""
    from psycopg2.extras import execute_batch

    if df.empty:
        return 0

    columns = list(df.columns)
    non_key_columns = [c for c in columns if c not in unique_keys]

    temp_table = f"_temp_{table_name}_{abs(id(df))}"

    try:
        conn = session.connection()
        raw_conn = conn.connection.driver_connection
        cursor = raw_conn.cursor()

        cursor.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {table_name} INCLUDING DEFAULTS) ON COMMIT DROP")

        col_clause = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        insert_sql = f"INSERT INTO {temp_table} ({col_clause}) VALUES ({placeholders})"

        records = [tuple(row) for row in df.values]
        execute_batch(cursor, insert_sql, records, page_size=batch_size)

        update_set = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key_columns])
        upsert_sql = f"""
            INSERT INTO {table_name} ({col_clause})
            SELECT {col_clause} FROM {temp_table}
            ON CONFLICT ({", ".join(unique_keys)}) DO UPDATE SET {update_set}
        """
        cursor.execute(upsert_sql)

        total = len(records)
        session.commit()

        logger.info(f"批量写入 {table_name} 表 {total} 条记录 (upsert via temp table)")
        return total

    except Exception as e:
        session.rollback()
        logger.warning(f"psycopg2 upsert 失败，回退到 SQLAlchemy 方式: {e}")
        return _bulk_upsert_sqlalchemy(session, table_name, df, unique_keys)


def query_df(
    session: Session,
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
        session: 数据库会话
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
    if columns:
        col_clause = ", ".join(columns)
        sql = f"SELECT {col_clause} FROM {table_name}"
    else:
        sql = f"SELECT * FROM {table_name}"

    params = {}
    if filters:
        where_clauses = []
        for key, value in filters.items():
            m = re.match(r'^(.+?)\s*([><=!]+)\s*$', key)
            if m:
                col_name = m.group(1).strip()
                op = m.group(2).strip()
                if op == '=':
                    op = '='
                elif op == '>':
                    op = '>'
                elif op == '<':
                    op = '<'
                elif op == '>=':
                    op = '>='
                elif op == '<=':
                    op = '<='
                elif op == '!=':
                    op = '!='
                safe_key = col_name.replace(' ', '_').replace('-', '_')
                placeholder = f":{safe_key}"
                where_clauses.append(f"{col_name} {op} {placeholder}")
                params[safe_key] = value
            elif isinstance(value, (list, tuple)):
                placeholders = ", ".join([f":{key}_{i}" for i in range(len(value))])
                where_clauses.append(f"{key} IN ({placeholders})")
                for i, v in enumerate(value):
                    params[f"{key}_{i}"] = v
            else:
                where_clauses.append(f"{key} = :{key}")
                params[key] = value
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

    if order_by:
        order_parts = []
        for part in order_by.split(','):
            part = part.strip()
            if not part:
                continue
            desc = part.startswith("-")
            col_name = part.lstrip("+-")
            order_parts.append(f"{col_name} {'DESC' if desc else 'ASC'}")
        if order_parts:
            sql += " ORDER BY " + ", ".join(order_parts)

    if limit:
        sql += f" LIMIT {limit}"
    if offset:
        sql += f" OFFSET {offset}"

    result = session.execute(text(sql), params)
    rows = result.fetchall()
    cols = result.keys()

    if not rows:
        return pd.DataFrame(columns=list(cols))

    return pd.DataFrame(rows, columns=list(cols))


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
    try:
        result = session.execute(text(sql), params or {})
        columns = result.keys()
        rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows, columns=columns)
    except Exception:
        session.rollback()
        return pd.DataFrame()


def count_records(
    session: Session,
    table_name: str,
    filters: Optional[Dict[str, Any]] = None
) -> int:
    """
    统计记录数

    Args:
        session: 数据库会话
        table_name: 表名
        filters: 过滤条件

    Returns:
        记录数
    """
    sql = f"SELECT COUNT(*) as cnt FROM {table_name}"
    params = {}

    if filters:
        where_clauses = []
        for key, value in filters.items():
            where_clauses.append(f"{key} = :{key}")
            params[key] = value
        sql += " WHERE " + " AND ".join(where_clauses)

    result = session.execute(text(sql), params)
    row = result.fetchone()
    return row[0] if row else 0


def table_exists(session: Session, table_name: str) -> bool:
    """
    检查表是否存在

    Args:
        session: 数据库会话
        table_name: 表名

    Returns:
        是否存在
    """
    if DATABASE_URL.startswith("sqlite"):
        sql = """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='table' AND name = :table_name
        """
    else:
        sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = :table_name
            )
        """
    result = session.execute(text(sql), {"table_name": table_name})
    return result.scalar() > 0


def get_table_columns(session: Session, table_name: str) -> List[str]:
    """
    获取表的列名列表

    Args:
        session: 数据库会话
        table_name: 表名

    Returns:
        列名列表
    """
    if DATABASE_URL.startswith("sqlite"):
        sql = f"PRAGMA table_info({table_name})"
        result = session.execute(text(sql))
        return [row[1] for row in result.fetchall()]
    else:
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

        print(f"表 stock_k_data 存在: {table_exists(session, 'stock_k_data')}")
        print(f"表 stock_amp_features 存在: {table_exists(session, 'stock_amp_features')}")