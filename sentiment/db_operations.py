#!/usr/bin/env python3
"""
Purpose: 舆情表 stock_sentiment_posts 的数据库操作封装
Inputs:   DataFrame 或查询参数
Outputs:  数据库读写结果
How to Run:
    本模块不直接运行，由 cli.py 调用
Examples:
    from sentiment.db_operations import init_sentiment_table, upsert_posts, get_latest_post_time
    init_sentiment_table()
    upsert_posts(df)
    latest = get_latest_post_time("SH688615")
Side Effects: 读写数据库 stock_sentiment_posts 表
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text

from datasource.database import get_session, bulk_upsert, get_engine
from datasource.models import SENTIMENT_POSTS_TABLE

logger = logging.getLogger(__name__)


def init_sentiment_table() -> None:
    """初始化舆情表（若不存在则创建）。"""
    with get_session() as session:
        session.execute(text(SENTIMENT_POSTS_TABLE))
        session.commit()
    logger.info("Table stock_sentiment_posts initialized")


def upsert_posts(df: pd.DataFrame) -> int:
    """将帖子 DataFrame 批量 upsert 到数据库。

    Args:
        df: 包含列 ts_code, post_id, author, post_time, title, content, link, source 的 DataFrame

    Returns:
        写入的行数
    """
    if df.empty:
        logger.info("No posts to upsert")
        return 0

    required = ["ts_code", "post_id", "post_time"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 确保 source 列存在
    if "source" not in df.columns:
        df["source"] = "xueqiu"

    with get_session() as session:
        count = bulk_upsert(
            session,
            "stock_sentiment_posts",
            df,
            unique_keys=["ts_code", "post_id", "source"],
        )
    logger.info("Upserted %d posts", count)
    return count


def _query_df(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    """使用 engine 执行查询并返回 DataFrame。"""
    from datasource.database import _result_to_df

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return _result_to_df(result)


def get_latest_post_time(ts_code: str, source: Optional[str] = None) -> Optional[datetime]:
    """查询某股票在数据库中的最新帖子时间。

    Args:
        ts_code: 股票代码
        source: 来源平台，如 xueqiu/eastmoney/tonghuashun，为 None 则不分平台

    Returns:
        最新帖子时间，若无记录则返回 None
    """
    if source:
        sql = """
        SELECT MAX(post_time) as latest_time
        FROM stock_sentiment_posts
        WHERE ts_code = :ts_code AND source = :source
        """
        params = {"ts_code": ts_code, "source": source}
    else:
        sql = """
        SELECT MAX(post_time) as latest_time
        FROM stock_sentiment_posts
        WHERE ts_code = :ts_code
        """
        params = {"ts_code": ts_code}
    df = _query_df(sql, params)
    if df.empty or df["latest_time"].isna().all():
        return None
    return pd.to_datetime(df["latest_time"].iloc[0]).to_pydatetime()


def get_posts(
    ts_code: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """按时间范围查询某股票的帖子。

    Args:
        ts_code: 股票代码
        start_time: 开始时间（包含）
        end_time: 结束时间（包含）
        source: 来源平台过滤

    Returns:
        帖子 DataFrame
    """
    conditions = ["ts_code = :ts_code"]
    params: dict = {"ts_code": ts_code}

    if start_time is not None:
        conditions.append("post_time >= :start_time")
        params["start_time"] = start_time
    if end_time is not None:
        conditions.append("post_time <= :end_time")
        params["end_time"] = end_time
    if source is not None:
        conditions.append("source = :source")
        params["source"] = source

    where_clause = " AND ".join(conditions)
    sql = f"""
    SELECT ts_code, post_id, author, post_time, title, content, link, source, created_at
    FROM stock_sentiment_posts
    WHERE {where_clause}
    ORDER BY post_time DESC
    """
    return _query_df(sql, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_sentiment_table()
    print("Table initialized successfully")
