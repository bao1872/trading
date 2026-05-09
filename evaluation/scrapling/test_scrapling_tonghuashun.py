#!/usr/bin/env python3
"""
Purpose: 用 Scrapling 实现同花顺帖子抓取，与 Selenium 版对比
Inputs:   股票代码 ts_code
Outputs:  终端打印帖子列表、与 Selenium 版对比
How to Run:
    python evaluation/scrapling/test_scrapling_tonghuashun.py
Examples:
    python evaluation/scrapling/test_scrapling_tonghuashun.py
Side Effects: 启动 Playwright 浏览器（headless），访问同花顺，不写入数据库
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


def safe_css_text(element, selector: str) -> str:
    """Scrapling 安全 CSS 文本提取。"""
    result = element.css(selector + "::text")
    return result.get("") or ""


def extract_post_scrapling(item) -> Optional[Dict]:
    """用 Scrapling Selector API 提取同花顺单条帖子。"""
    result: Dict = {}

    # 标题与链接
    title_el = item.css(".item-title") or item.css(".title")
    if title_el:
        result["title"] = title_el[0].text.strip()
        href = title_el[0].attrib.get("href", "")
        result["link"] = href if href.startswith("http") else f"https://basic.10jqka.com.cn{href}"
    else:
        result["title"] = ""
        result["link"] = ""

    # post_id
    if result["link"]:
        parts = result["link"].rstrip("/").split("/")
        result["post_id"] = parts[-1].replace(".html", "") if parts else ""
    else:
        result["post_id"] = ""

    if not result["post_id"]:
        return None

    # 作者
    result["author"] = safe_css_text(item, ".author-name") or safe_css_text(item, ".author")

    # 时间
    result["time_text"] = safe_css_text(item, ".item-time") or safe_css_text(item, ".time")
    result["post_time"] = parse_relative_time(result["time_text"])

    # 内容摘要
    result["content"] = safe_css_text(item, ".item-content") or safe_css_text(item, ".content")

    result["source"] = "tonghuashun"
    return result


def fetch_posts_scrapling(
    ts_code: str,
    start_time: datetime,
    max_scrolls: int = 3,
    scroll_delay: float = 2.0,
) -> List[Dict]:
    """用 Scrapling DynamicFetcher 抓取同花顺帖子（滚动加载）。"""
    from scrapling.fetchers import DynamicFetcher

    all_posts: List[Dict] = []
    code = _extract_code(ts_code)

    url = f"https://stockpage.10jqka.com.cn/{code}/"
    logger.info("Fetching: %s", url)

    t0 = time.time()
    page = DynamicFetcher.fetch(url, headless=True, network_idle=True, timeout=60000)
    logger.info("Initial load: %.1fs, title=%s", time.time() - t0, page.css("title::text").get(""))

    for scroll_idx in range(max_scrolls):
        items = (
            page.css(".list-item")
            or page.css(".item")
            or page.css(".post-item")
        )
        logger.info("Scroll %d: found %d items", scroll_idx, len(items))

        if not items:
            break

        batch_posts = []
        for item in items:
            post = extract_post_scrapling(item)
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

        # 滚动（同花顺用 DynamicFetcher，不需要 Stealthy）
        time.sleep(scroll_delay)
        page = DynamicFetcher.fetch(url, headless=True, network_idle=True, timeout=60000)

    logger.info("Total posts: %d, total time: %.1fs", len(all_posts), time.time() - t0)
    return all_posts


def compare_with_selenium():
    from sentiment.tonghuashun_scraper import fetch_posts as fetch_ths_selenium

    start = datetime.now() - timedelta(days=7)
    logger.info("Running Selenium version for comparison...")

    t0 = time.time()
    sel_posts = fetch_ths_selenium("SH688615", start, max_scrolls=3)
    sel_elapsed = time.time() - t0
    logger.info("Selenium: %d posts in %.1fs", len(sel_posts), sel_elapsed)

    return sel_posts, sel_elapsed


def main():
    print("=" * 60)
    print("同花顺 Scrapling vs Selenium 对比测试")
    print("=" * 60)

    ts_code = "SH688615"
    start = datetime.now() - timedelta(days=30)

    print("\n--- Scrapling DynamicFetcher ---")
    t0 = time.time()
    posts = fetch_posts_scrapling(ts_code, start, max_scrolls=3)
    sc_elapsed = time.time() - t0
    print(f"Scrapling: {len(posts)} posts in {sc_elapsed:.1f}s")

    if posts:
        print("\n前5条:")
        for p in posts[:5]:
            print(f"  {p['post_time']} | {p['author']:12s} | {p['title'][:60]}")

        print("\n字段验证:")
        p = posts[0]
        for k in ['ts_code','post_id','author','post_time','title','content','link','source']:
            print(f"  {k}: {'OK' if k in p else 'MISSING'} = {str(p.get(k))[:60]}")

    # Selenium 对比
    print("\n--- Selenium 版对比 ---")
    sel_posts, sel_elapsed = compare_with_selenium()
    print(f"Selenium: {len(sel_posts)} posts in {sel_elapsed:.1f}s")

    print("\n" + "=" * 60)
    print(f"同花顺平台结论:")
    print(f"  Scrapling: {len(posts)} posts, {sc_elapsed:.1f}s")
    print(f"  Selenium:  {len(sel_posts)} posts, {sel_elapsed:.1f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())