#!/usr/bin/env python3
"""
Purpose: 高质量评论博主发现与分析：基于发帖频率与内容深度评分，展示博主时间线
Inputs:   股票代码、时间窗口、来源平台
Outputs:  博主排名 DataFrame、博主帖子时间线
How to Run:
    本模块由 blogger_cli.py 调用，也可通过 __main__ 自测
Examples:
    from sentiment.blogger_analyzer import analyze_bloggers, show_blogger_timeline
    rankings = analyze_bloggers("SH688615", months=3, sources=["xueqiu", "eastmoney"])
    timeline = show_blogger_timeline("SH688615", "某博主", months=3, sources=["xueqiu"])
Side Effects: 读数据库 stock_sentiment_posts 表，不写入
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from sentiment.db_operations import get_posts

logger = logging.getLogger(__name__)

DEFAULT_W_FREQ = 0.6
DEFAULT_W_CONTENT = 0.4


def get_blogger_stats(
    ts_code: str,
    start_time: datetime,
    sources: Optional[List[str]] = None,
) -> pd.DataFrame:
    """查询指定股票时间窗口内各博主的基础统计。

    Args:
        ts_code: 股票代码，如 SH688615
        start_time: 时间窗口起始时间
        sources: 来源平台列表，如 ["xueqiu", "eastmoney"]，None 表示全部

    Returns:
        DataFrame，列：author, post_count, avg_content_len,
        first_post, last_post, active_days, sources
    """
    if sources is None:
        sources = ["xueqiu", "eastmoney"]

    all_posts = []
    for src in sources:
        df = get_posts(ts_code, start_time=start_time, source=src)
        if not df.empty:
            all_posts.append(df)

    if not all_posts:
        logger.warning("No posts found for %s from %s", ts_code, sources)
        return pd.DataFrame(
            columns=["author", "post_count", "avg_content_len", "first_post",
                     "last_post", "active_days", "sources"]
        )

    df_all = pd.concat(all_posts, ignore_index=True)
    if df_all.empty:
        return pd.DataFrame(
            columns=["author", "post_count", "avg_content_len", "first_post",
                     "last_post", "active_days", "sources"]
        )

    df_all["content_len"] = df_all["content"].fillna("").str.len()

    stats = df_all.groupby("author").agg(
        post_count=("post_id", "count"),
        avg_content_len=("content_len", "mean"),
        first_post=("post_time", "min"),
        last_post=("post_time", "max"),
        sources=("source", lambda x: ",".join(sorted(set(x)))),
    ).reset_index()

    stats["active_days"] = (
        (stats["last_post"] - stats["first_post"]).dt.total_seconds()
        / 86400.0 + 1
    )
    stats["daily_freq"] = stats["post_count"] / stats["active_days"]

    return stats


def score_bloggers(
    stats_df: pd.DataFrame,
    w_freq: float = DEFAULT_W_FREQ,
    w_content: float = DEFAULT_W_CONTENT,
) -> pd.DataFrame:
    """基于频率与内容深度计算综合质量分，返回排名。

    评分公式：
        freq_score = ln(1+post_count) / ln(1+max_post_count) * 100
        content_score = ln(1+avg_content_len) / ln(1+max_avg_len) * 100
        total = w_freq * freq_score + w_content * content_score

    Args:
        stats_df: get_blogger_stats 的输出
        w_freq: 频率权重，默认 0.6
        w_content: 内容深度权重，默认 0.4

    Returns:
        含 score, freq_score, content_score, rank 列的排名 DataFrame
    """
    if stats_df.empty:
        return stats_df

    df = stats_df.copy()

    max_posts = df["post_count"].max()
    max_len = df["avg_content_len"].max()

    if max_posts > 0:
        df["freq_score"] = (
            np.log1p(df["post_count"]) / np.log1p(max_posts) * 100
        )
    else:
        df["freq_score"] = 0.0

    if max_len > 0:
        df["content_score"] = (
            np.log1p(df["avg_content_len"]) / np.log1p(max_len) * 100
        )
    else:
        df["content_score"] = 0.0

    df["score"] = w_freq * df["freq_score"] + w_content * df["content_score"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def analyze_bloggers(
    ts_code: str,
    months: int = 3,
    sources: Optional[List[str]] = None,
    top_n: int = 10,
    w_freq: float = DEFAULT_W_FREQ,
    w_content: float = DEFAULT_W_CONTENT,
) -> pd.DataFrame:
    """一站式分析：查统计 → 评分 → 返回 Top N。

    Args:
        ts_code: 股票代码
        months: 时间窗口月数
        sources: 来源平台列表
        top_n: 返回前 N 名
        w_freq: 频率权重
        w_content: 内容深度权重

    Returns:
        含 score, freq_score, content_score, rank 的 Top N 博主 DataFrame
    """
    start_time = datetime.now() - timedelta(days=months * 30)
    stats = get_blogger_stats(ts_code, start_time, sources)
    if stats.empty:
        print(f"未找到 {ts_code} 近 {months} 个月内的帖子记录")
        return stats

    scored = score_bloggers(stats, w_freq=w_freq, w_content=w_content)
    return scored.head(top_n).copy()


def show_blogger_timeline(
    ts_code: str,
    author: str,
    months: int = 3,
    sources: Optional[List[str]] = None,
) -> pd.DataFrame:
    """展示指定博主在时间窗口内的所有帖子，按时间升序排列。

    Args:
        ts_code: 股票代码
        author: 博主用户名（精确匹配）
        months: 时间窗口月数
        sources: 来源平台列表

    Returns:
        帖子 DataFrame，按 post_time 升序
    """
    if sources is None:
        sources = ["xueqiu", "eastmoney"]

    start_time = datetime.now() - timedelta(days=months * 30)
    all_posts = []
    for src in sources:
        df = get_posts(ts_code, start_time=start_time, source=src)
        if not df.empty:
            df = df[df["author"] == author]
            if not df.empty:
                all_posts.append(df)

    if not all_posts:
        logger.warning("No posts found for author %s on %s", author, ts_code)
        return pd.DataFrame()

    df_all = pd.concat(all_posts, ignore_index=True)
    df_all = df_all.sort_values("post_time").reset_index(drop=True)
    return df_all[["post_time", "title", "content", "link", "source"]]


def format_blogger_ranking(scored_df: pd.DataFrame) -> str:
    """格式化博主排名为可读字符串。"""
    if scored_df.empty:
        return "无数据"

    lines = []
    lines.append(f"{'排名':<5} {'博主':<20} {'帖子数':<8} {'内容均长':<10} {'质量分':<8} {'平台':<20}")
    lines.append("-" * 75)
    for _, row in scored_df.iterrows():
        lines.append(
            f"{row['rank']:<5} {str(row['author']):<20} "
            f"{int(row['post_count']):<8} {int(row['avg_content_len']):<10} "
            f"{row['score']:<8.1f} {str(row['sources']):<20}"
        )
    return "\n".join(lines)


def format_timeline(timeline_df: pd.DataFrame, author: str) -> str:
    """格式化博主时间线为可读字符串。"""
    if timeline_df.empty:
        return f"博主 {author} 在指定时间窗口内无帖子记录"

    lines = [f"\n博主 {author} 的帖子时间线（共 {len(timeline_df)} 条）：\n"]
    for idx, (_, row) in enumerate(timeline_df.iterrows(), 1):
        post_time = str(row["post_time"])[:16]
        title = str(row["title"])[:80]
        source = row["source"]
        link = row["link"]
        lines.append(f"[{idx}] {post_time} [{source}]")
        lines.append(f"    标题: {title}")
        content = str(row["content"])[:200]
        if content:
            lines.append(f"    内容: {content}")
        if link:
            lines.append(f"    链接: {link}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("=== 自测: 博主分析模块 ===\n")

    ts_code = "SH688615"
    months = 3
    sources = ["xueqiu", "eastmoney"]

    print(f"1) 分析 {ts_code} 近{months}个月高质量博主 Top 5:")
    rankings = analyze_bloggers(ts_code, months=months, sources=sources, top_n=5)
    print(format_blogger_ranking(rankings))

    if not rankings.empty:
        first_author = rankings.iloc[0]["author"]
        print(f"\n2) 展示博主 [{first_author}] 的时间线:")
        timeline = show_blogger_timeline(ts_code, first_author, months=months, sources=sources)
        print(format_timeline(timeline, first_author))
    else:
        print("\n无博主数据，请先运行 sentiment.cli 抓取帖子系统录入数据")
