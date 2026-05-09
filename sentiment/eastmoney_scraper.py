#!/usr/bin/env python3
"""
Purpose: 东方财富股吧帖子抓取（Scrapling DynamicFetcher 版，直接分页URL）
Inputs:   ts_code (如 SH688615), start_time (datetime), max_pages, page_delay
Outputs:  List[Dict] — 8 字段（ts_code/post_id/author/post_time/title/content/link/source）
How to Run:
    PYTHONPATH=/root/trading python sentiment/eastmoney_scraper.py
Examples:
    from sentiment.eastmoney_scraper import fetch_posts
    posts = fetch_posts("SH688615", datetime.now()-timedelta(days=7), max_pages=3)
Side Effects: 启动 Playwright headless 浏览器（DynamicFetcher），访问东方财富股吧，不写入数据库
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from sentiment.scraper_utils import parse_relative_time

logger = logging.getLogger(__name__)


def _extract_code(ts_code: str) -> str:
    return "".join(filter(str.isdigit, ts_code))


def fetch_posts(
    ts_code: str,
    start_time: datetime,
    max_pages: int = 10,
    page_delay: float = 0.5,
) -> List[Dict]:
    """抓取东方财富股吧帖子列表（分页）。

    Args:
        ts_code: 股票代码，如 SH688615
        start_time: 只返回此时间之后的帖子
        max_pages: 最大翻页数
        page_delay: 翻页间隔秒数

    Returns:
        List[Dict]: 帖子列表，8 字段（ts_code/post_id/author/post_time/title/content/link/source）
    """
    from scrapling.fetchers import DynamicFetcher

    all_posts: List[Dict] = []
    code = _extract_code(ts_code)
    seen_ids: set = set()

    t0 = time.time()
    for page_n in range(1, max_pages + 1):
        url = (
            f"https://guba.eastmoney.com/list,{code}.html"
            if page_n == 1
            else f"https://guba.eastmoney.com/list,{code}_{page_n}.html"
        )

        try:
            page = DynamicFetcher.fetch(url, headless=True, network_idle=True, timeout=60000)
        except Exception as e:
            logger.error("eastmoney %s page %d fetch failed: %s", ts_code, page_n, e)
            break

        trs = page.css("tr")
        rows = []
        for tr in trs:
            tds = tr.css("td")
            if len(tds) >= 5:
                text0 = tds[0].get_all_text().strip()
                if text0 and text0 != "阅读":
                    rows.append(tr)

        if not rows:
            logger.info("eastmoney %s page %d: no rows, stopping", ts_code, page_n)
            break

        logger.info("eastmoney %s page %d: %d rows", ts_code, page_n, len(rows))

        batch_posts = []
        for row in rows:
            tds = row.css("td")
            post: Dict = {}
            post["source"] = "eastmoney"
            post["ts_code"] = ts_code

            post["read_count"] = tds[0].get_all_text().strip()
            post["comment_count"] = tds[1].get_all_text().strip()

            title_els = tds[2].css("a")
            if title_els:
                a_tag = title_els[0]
                post["title"] = a_tag.get_all_text().strip().replace("资讯\n", "")
                href = a_tag.attrib.get("href", "")
                post["link"] = href if href.startswith("http") else f"https://guba.eastmoney.com{href}"
            else:
                post["title"] = tds[2].get_all_text().strip().replace("资讯\n", "")
                post["link"] = ""

            if post["link"]:
                parts = post["link"].rstrip("/").split(",")
                post["post_id"] = parts[-1].replace(".html", "") if parts else ""
            else:
                post["post_id"] = ""

            if not post["post_id"]:
                continue

            post["author"] = tds[3].get_all_text().strip()
            time_text = tds[4].get_all_text().strip()
            post["time_text"] = time_text
            post["post_time"] = parse_relative_time(time_text)
            post["content"] = ""

            batch_posts.append(post)

        batch_posts.sort(key=lambda x: x["post_time"], reverse=True)
        oldest = batch_posts[-1]["post_time"]
        logger.info("  %s ~ %s", batch_posts[0]["post_time"], oldest)

        for post in batch_posts:
            pid = post["post_id"]
            if pid not in seen_ids and post["post_time"] >= start_time:
                all_posts.append(post)
                seen_ids.add(pid)

        if oldest < start_time:
            logger.info("eastmoney %s: oldest %s < start %s, stopping", ts_code, oldest, start_time)
            break

        time.sleep(page_delay)

    all_posts.sort(key=lambda x: x["post_time"], reverse=True)
    logger.info("eastmoney %s: %d posts in %.1fs (%d pages)", ts_code, len(all_posts), time.time() - t0, page_n)
    return all_posts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ts_code = "SH688615"
    start = datetime.now() - timedelta(days=30)

    print(f"Fetching eastmoney posts for {ts_code} since {start}")
    posts = fetch_posts(ts_code, start, max_pages=2)

    print(f"Total: {len(posts)} posts")
    for p in posts[:5]:
        print(f"  {p['post_time']} | {p['author'][:12]:12s} | {p.get('title','')[:50]}")