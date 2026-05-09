#!/usr/bin/env python3
"""
Purpose: 用 Scrapling StealthyFetcher 实现雪球帖子抓取，与 Selenium 版对比
Inputs:   股票代码 ts_code
Outputs:  终端打印帖子列表、与 Selenium 版对比结果
How to Run:
    python evaluation/scrapling/test_scrapling_xueqiu.py
Examples:
    python evaluation/scrapling/test_scrapling_xueqiu.py
Side Effects: 启动 Playwright 浏览器（headless），访问雪球网，不写入数据库
"""

import time
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _extract_code(ts_code: str) -> str:
    return "".join(filter(str.isdigit, ts_code))


def parse_relative_time(text: str) -> datetime:
    """中文相对时间解析（从 scraper_utils.py 引用）。"""
    import re
    now = datetime.now()
    text = text.strip()

    m = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", text)
    if m:
        month, day, hour, minute = map(int, m.groups())
        dt = datetime(now.year, month, day, hour, minute)
        if dt > now + timedelta(days=1):
            dt = dt.replace(year=now.year - 1)
        return dt

    m = re.match(r"昨天\s+(\d{2}):(\d{2})", text)
    if m:
        hour, minute = map(int, m.groups())
        return (now - timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)

    m = re.match(r"今天\s+(\d{2}):(\d{2})", text)
    if m:
        hour, minute = map(int, m.groups())
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    m = re.match(r"(\d{2}):(\d{2})", text)
    if m:
        hour, minute = map(int, m.groups())
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", text)
    if m:
        year, month, day, hour, minute = map(int, m.groups())
        return datetime(year, month, day, hour, minute)

    m = re.match(r"刚刚", text)
    if m:
        return now

    m = re.match(r"(\d+)\s*分钟前", text)
    if m:
        return now - timedelta(minutes=int(m.group(1)))

    m = re.match(r"(\d+)\s*小时前", text)
    if m:
        return now - timedelta(hours=int(m.group(1)))

    return now


def extract_post_scrapling(article) -> Optional[Dict]:
    """用 Scrapling Selector API 提取雪球单条 article 的帖子信息。"""
    result: Dict = {}

    # 作者: .user-name
    result["author"] = article.css(".user-name::text").get("")

    # 时间: .date-and-source
    time_text = article.css(".date-and-source::text").get("")
    if not time_text:
        return None
    time_text = time_text.split("·")[0].strip()
    result["time_text"] = time_text
    result["post_time"] = parse_relative_time(time_text)

    # 链接与 post_id: a[data-id]
    links = article.css("a[data-id]")
    if links:
        data_id = links[0].attrib.get("data-id", "")
        result["post_id"] = data_id
        result["link"] = f"https://xueqiu.com/{data_id}" if data_id else ""
    else:
        # fallback: .date-and-source a
        link_els = article.css(".date-and-source a")
        if link_els:
            href = link_els[0].attrib.get("href", "")
            result["link"] = href
            parts = href.rstrip("/").split("/")
            result["post_id"] = parts[-1] if parts else ""
        else:
            return None

    if not result.get("post_id"):
        return None

    # 内容: .content--description
    content_el = article.css(".content--description::text")
    full_text = "".join(content_el.getall()).strip() or article.css(".content::text").get("")

    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    result["title"] = lines[0] if lines else ""
    result["content"] = "\n".join(lines[1:]) if len(lines) > 1 else ""

    result["source"] = "xueqiu"
    return result


def fetch_posts_scrapling(
    ts_code: str,
    start_time: datetime,
    max_scrolls: int = 5,
    scroll_delay: float = 2.0,
) -> List[Dict]:
    """用 Scrapling StealthySession 抓取雪球帖子（含滚动加载）。

    Args:
        ts_code: 如 SH688615
        start_time: 开始时间
        max_scrolls: 最大滚动次数
        scroll_delay: 滚动间隔秒数

    Returns:
        帖子 dict 列表
    """
    from scrapling.fetchers import StealthyFetcher

    all_posts: List[Dict] = []

    url = f"https://xueqiu.com/S/{ts_code}"
    logger.info("Scrapling fetch: %s", url)

    t0 = time.time()
    page = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=60000)
    logger.info("Initial load: %.1fs, title=%.60s", time.time() - t0, page.css("title::text").get(""))

    for scroll_idx in range(max_scrolls):
        articles = page.css("article")
        logger.info("Scroll %d: found %d articles", scroll_idx, len(articles))

        batch_posts = []
        for art in articles:
            post = extract_post_scrapling(art)
            if post is None:
                continue
            post["ts_code"] = ts_code
            batch_posts.append(post)

        if not batch_posts:
            break

        batch_posts.sort(key=lambda x: x["post_time"], reverse=True)
        oldest = batch_posts[-1]["post_time"]
        logger.info("  Batch: %s ~ %s", batch_posts[0]["post_time"], oldest)

        seen_ids = {p["post_id"] for p in all_posts}
        for post in batch_posts:
            if post["post_id"] not in seen_ids and post["post_time"] >= start_time:
                all_posts.append(post)

        if oldest < start_time:
            logger.info("Oldest post %s < %s, stopping", oldest, start_time)
            break

        # 滚动加载 — 用 Scrapling page 不能直接 execute_script
        # 改为每次重新请求或者用 session
        logger.info("Re-fetching for next scroll batch...")
        time.sleep(scroll_delay)
        # Note: Scrapling 的 page 对象在一次 fetch 后内容固定，
        # 要加载更多需要重新 fetch（或使用 session 并手动执行 JS）
        # 这里用重新 fetch 模拟滚动
        page = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=60000)

    logger.info("Total posts: %d, total time: %.1fs", len(all_posts), time.time() - t0)
    return all_posts


def fetch_posts_scrapling_session(
    ts_code: str,
    start_time: datetime,
    max_scrolls: int = 5,
    scroll_delay: float = 2.0,
) -> List[Dict]:
    """用 Scrapling StealthySession 抓取雪球帖子，page_action 实现原生 JS 滚动。"""
    from scrapling.fetchers import StealthySession

    all_posts: List[Dict] = []
    url = f"https://xueqiu.com/S/{ts_code}"
    logger.info("Scrapling session fetch: %s", url)

    t0 = time.time()

    def scroll_and_collect(pw_page):
        """page_action 回调：执行滚动并等待内容加载。"""
        pw_page.wait_for_timeout(int(scroll_delay * 1000))
        for _ in range(max_scrolls):
            pw_page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            pw_page.wait_for_timeout(int(scroll_delay * 1000))

    with StealthySession(headless=True) as session:
        page = session.fetch(url, network_idle=True, page_action=scroll_and_collect, timeout=60000)
        logger.info("Session load+scroll: %.1fs", time.time() - t0)

        articles = page.css("article")
        logger.info("Found %d articles total", len(articles))

        for art in articles:
            post = extract_post_scrapling(art)
            if post is None:
                continue
            post["ts_code"] = ts_code
            if post["post_time"] >= start_time:
                all_posts.append(post)

        all_posts.sort(key=lambda x: x["post_time"], reverse=True)

    logger.info("Total posts: %d, total time: %.1fs", len(all_posts), time.time() - t0)
    return all_posts


def compare_with_selenium(scrapling_posts: List[Dict]):
    """用 Selenium 版抓同样数据做对比。"""
    from sentiment.xueqiu_scraper import fetch_posts as fetch_xq_selenium

    start = datetime.now() - timedelta(days=7)
    logger.info("Running Selenium version for comparison...")

    t0 = time.time()
    sel_posts = fetch_xq_selenium("SH688615", start, max_scrolls=3)
    sel_elapsed = time.time() - t0

    logger.info("Selenium: %d posts in %.1fs", len(sel_posts), sel_elapsed)

    return sel_posts, sel_elapsed


def main():
    print("=" * 60)
    print("雪球 Scrapling vs Selenium 对比测试")
    print("=" * 60)

    ts_code = "SH688615"
    start = datetime.now() - timedelta(days=7)

    # Scrapling Session 版（推荐方式）
    print("\n--- Scrapling StealthySession ---")
    t0 = time.time()
    posts = fetch_posts_scrapling_session(ts_code, start, max_scrolls=3)
    sc_elapsed = time.time() - t0

    print(f"\nScrapling: {len(posts)} posts in {sc_elapsed:.1f}s")
    print("\n前5条:")
    for p in posts[:5]:
        print(f"  {p['post_time']} | {p['author']:12s} | {p['title'][:60]}")

    print(f"\n--- 字段对比 ---")
    if posts:
        p = posts[0]
        print(f"  ts_code={p.get('ts_code')}, post_id={p.get('post_id')}")
        print(f"  author={p.get('author')}, source={p.get('source')}")
        print(f"  time_text={p.get('time_text')}, post_time={p.get('post_time')}")
        print(f"  link={p.get('link')}")
        print(f"  title={p.get('title')[:80]}")
        print(f"  content={p.get('content')[:100]}")
        print(f"  Fields OK: {all(k in p for k in ['ts_code','post_id','author','post_time','title','content','link','source'])}")

    # Selenium 对比
    print("\n--- Selenium 版对比 ---")
    sel_posts, sel_elapsed = compare_with_selenium(posts)
    print(f"\nSelenium: {len(sel_posts)} posts in {sel_elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("雪球平台结论:")
    print(f"  Scrapling (Session): {len(posts)} posts, {sc_elapsed:.1f}s")
    print(f"  Selenium:            {len(sel_posts)} posts, {sel_elapsed:.1f}s")
    print("  关键: 必须用 StealthyFetcher/StealthySession 绕过反爬")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())