#!/usr/bin/env python3
"""
Purpose: 舆情抓取命令行入口，支持雪球/东方财富/同花顺三平台全量/增量/dry-run
         （雪球和东方财富已迁移到 Scrapling，同花顺页面结构异常待修复）
Inputs:   命令行参数
Outputs:  终端打印进度与结果，数据写入数据库
How to Run:
    python -m sentiment.cli --ts-code SH688615 --source xueqiu --start-time 2026-04-01
    python -m sentiment.cli --ts-code SH688615 --source eastmoney --incremental
    python -m sentiment.cli --ts-code SH688615 --source all --start-time 2026-04-01 --dry-run
Examples:
    # 抓取合合信息最近3天帖子（雪球）
    python -m sentiment.cli --ts-code SH688615 --source xueqiu --start-time 2026-05-03

    # 增量更新东方财富
    python -m sentiment.cli --ts-code SH688615 --source eastmoney --incremental
Side Effects: 读写数据库 stock_sentiment_posts 表；启动 Playwright 浏览器
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from sentiment.xueqiu_scraper import fetch_posts as fetch_xueqiu
from sentiment.eastmoney_scraper import fetch_posts as fetch_eastmoney
from sentiment.tonghuashun_scraper import fetch_posts as fetch_tonghuashun
from sentiment.db_operations import init_sentiment_table, upsert_posts, get_latest_post_time

logger = logging.getLogger(__name__)

SCRAPER_MAP = {
    "xueqiu": fetch_xueqiu,
    "eastmoney": fetch_eastmoney,
    "tonghuashun": fetch_tonghuashun,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="舆情数据抓取工具（雪球/东方财富/同花顺）")
    parser.add_argument(
        "--ts-code",
        required=True,
        help="股票代码，支持多个用逗号分隔，如 SH688615,SZ000001",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="xueqiu",
        choices=["xueqiu", "eastmoney", "tonghuashun", "all"],
        help="来源平台：xueqiu/eastmoney/tonghuashun/all（tonghuashun 页面结构异常，已知返回空）",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="开始时间，格式 YYYY-MM-DD，默认7天前",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量模式：从数据库中该股票最新帖子时间开始抓取",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印不写入数据库",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="东方财富最大翻页数，默认5（每页约60s）",
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=0.5,
        help="东方财富翻页间隔秒数，默认0.5",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=10,
        help="雪球/同花顺最大滚动次数，默认10",
    )
    parser.add_argument(
        "--scroll-delay",
        type=float,
        default=2.0,
        help="雪球每次滚动等待秒数，默认2.0",
    )
    return parser.parse_args()


def resolve_start_time(ts_code: str, source: str, start_time_str: Optional[str], incremental: bool) -> datetime:
    """解析开始时间，支持增量模式。"""
    if incremental:
        latest = get_latest_post_time(ts_code, source)
        if latest is not None:
            logger.info("Incremental mode for %s/%s: start from %s", ts_code, source, latest)
            return latest
        logger.info("No existing data for %s/%s, fallback to 7 days ago", ts_code, source)
        return datetime.now() - timedelta(days=7)

    if start_time_str:
        return datetime.strptime(start_time_str, "%Y-%m-%d")

    return datetime.now() - timedelta(days=7)


def scrape_single_stock(
    ts_code: str,
    source: str,
    start_time: datetime,
    args: argparse.Namespace,
    dry_run: bool,
) -> int:
    """抓取单个股票单个平台并入库，返回写入行数。"""
    logger.info("Scraping %s/%s from %s", ts_code, source, start_time)
    fetch_fn = SCRAPER_MAP[source]

    if source == "eastmoney":
        posts = fetch_fn(ts_code, start_time, max_pages=args.max_pages, page_delay=args.page_delay)
    else:
        posts = fetch_fn(ts_code, start_time, max_scrolls=args.max_scrolls, scroll_delay=args.scroll_delay)

    if not posts:
        logger.info("No posts found for %s/%s", ts_code, source)
        return 0

    df = pd.DataFrame(posts)
    df = df[["ts_code", "post_id", "author", "post_time", "title", "content", "link", "source"]]

    if dry_run:
        logger.info("[DRY-RUN] Would upsert %d posts for %s/%s", len(df), ts_code, source)
        print(df.head().to_string())
        return 0

    return upsert_posts(df)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.dry_run:
        init_sentiment_table()

    ts_codes = [code.strip() for code in args.ts_code.split(",")]
    sources = list(SCRAPER_MAP.keys()) if args.source == "all" else [args.source]
    total = 0

    for ts_code in ts_codes:
        for source in sources:
            start_time = resolve_start_time(ts_code, source, args.start_time, args.incremental)
            try:
                count = scrape_single_stock(ts_code, source, start_time, args, args.dry_run)
                total += count
                logger.info("Stock %s/%s done: %d posts", ts_code, source, count)
            except Exception as e:
                logger.error("Stock %s/%s FAILED: %s", ts_code, source, e)

    logger.info("All done. Total posts: %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())