#!/usr/bin/env python3
"""
Purpose: 雪球帖子抓取（Scrapling StealthySession 版，绕过反爬）
Inputs:   ts_code (如 SH688615), start_time (datetime), max_scrolls, scroll_delay
Outputs:  List[Dict] — 8 字段（ts_code/post_id/author/post_time/title/content/link/source）
How to Run:
    PYTHONPATH=/root/trading python sentiment/xueqiu_scraper.py
Examples:
    from sentiment.xueqiu_scraper import fetch_posts
    posts = fetch_posts("SH688615", datetime.now()-timedelta(days=7), max_scrolls=3)
Side Effects: 启动 Playwright headless 浏览器（StealthySession），访问雪球网，不写入数据库
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from sentiment.scraper_utils import parse_relative_time

logger = logging.getLogger(__name__)


def fetch_posts(
    ts_code: str,
    start_time: datetime,
    max_scrolls: int = 10,
    scroll_delay: float = 2.0,
) -> List[Dict]:
    """抓取雪球股票讨论页的帖子列表（含滚动加载）。

    Args:
        ts_code: 股票代码，如 SH688615
        start_time: 只返回此时间之后的帖子
        max_scrolls: page_action 内滚动次数
        scroll_delay: 两次滚动之间的等待秒数

    Returns:
        List[Dict]: 帖子列表，8 字段（ts_code/post_id/author/post_time/title/content/link/source）
    """
    from scrapling.fetchers import StealthySession

    all_posts: List[Dict] = []
    url = f"https://xueqiu.com/S/{ts_code}"

    def scroll(pw_page):
        pw_page.wait_for_timeout(int(scroll_delay * 1000))
        for _ in range(max_scrolls):
            pw_page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            pw_page.wait_for_timeout(int(scroll_delay * 1000))

    t0 = time.time()
    with StealthySession(headless=True) as session:
        page = session.fetch(url, network_idle=True, page_action=scroll, timeout=60000)
        logger.info("xueqiu %s loaded+scrolled in %.1fs", ts_code, time.time() - t0)

        articles = page.css("article")
        logger.info("xueqiu %s: %d articles found", ts_code, len(articles))

        for art in articles:
            post: Dict = {}
            post["source"] = "xueqiu"
            post["ts_code"] = ts_code

            post["author"] = art.css(".user-name::text").get("")

            time_text = art.css(".date-and-source::text").get("")
            if not time_text:
                continue
            time_text = time_text.split("·")[0].strip()
            post["time_text"] = time_text
            post["post_time"] = parse_relative_time(time_text)

            links = art.css("a[data-id]")
            if links:
                data_id = links[0].attrib.get("data-id", "")
                post["post_id"] = data_id
                post["link"] = f"https://xueqiu.com/{data_id}"
            else:
                link_els = art.css(".date-and-source a")
                if link_els:
                    href = link_els[0].attrib.get("href", "")
                    post["link"] = href
                    parts = href.rstrip("/").split("/")
                    post["post_id"] = parts[-1] if parts else ""
                else:
                    continue

            if not post.get("post_id"):
                continue

            cd = art.css(".content--description")
            if cd:
                post["content"] = cd[0].get_all_text().strip()
            else:
                post["content"] = ""

            lines = [line.strip() for line in post["content"].splitlines() if line.strip()]
            post["title"] = lines[0] if lines else ""

            if post["post_time"] >= start_time:
                all_posts.append(post)

    all_posts.sort(key=lambda x: x["post_time"], reverse=True)
    logger.info("xueqiu %s: %d valid posts (time filter from %s)", ts_code, len(all_posts), start_time)
    return all_posts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ts_code = "SH688615"
    start = datetime.now() - timedelta(days=30)

    print(f"Fetching xueqiu posts for {ts_code} since {start}")
    posts = fetch_posts(ts_code, start, max_scrolls=3)

    print(f"Total: {len(posts)} posts")
    for p in posts[:5]:
        print(f"  {p['post_time']} | {p['author'][:12]:12s} | {p.get('title','')[:50]}")