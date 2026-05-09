#!/usr/bin/env python3
"""
Purpose: 用 Scrapling 实现东方财富股吧帖子抓取，与 Selenium 版对比
Inputs:   股票代码 ts_code
Outputs:  终端打印帖子列表、与 Selenium 版对比
How to Run:
    python evaluation/scrapling/test_scrapling_eastmoney.py
Examples:
    python evaluation/scrapling/test_scrapling_eastmoney.py
Side Effects: 启动 Playwright 浏览器（headless），访问东方财富，不写入数据库
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


def extract_post_scrapling(row) -> Optional[Dict]:
    """用 Scrapling Selector API 提取东方财富单行的帖子信息。

    东方财富帖子列表结构:
      <tr>
        <td>阅读数</td>
        <td>评论数</td>
        <td><a>标题</a></td>    — 含链接
        <td>作者</td>
        <td>时间</td>
      </tr>
    """
    tds = row.css("td")
    if len(tds) < 5:
        return None

    result: Dict = {}
    result["read_count"] = tds[0].text.strip()
    result["comment_count"] = tds[1].text.strip()

    # 标题与链接（tds[2] 内的 a 标签）
    title_els = tds[2].css("a")
    if title_els:
        a_tag = title_els[0]
        result["title"] = a_tag.text.strip()
        href = a_tag.attrib.get("href", "")
        if href.startswith("http"):
            result["link"] = href
        else:
            result["link"] = f"https://guba.eastmoney.com{href}"
    else:
        result["title"] = tds[2].text.strip()
        result["link"] = ""

    # post_id
    if result["link"]:
        parts = result["link"].rstrip("/").split(",")
        result["post_id"] = parts[-1].replace(".html", "") if parts else ""
    else:
        result["post_id"] = ""

    if not result["post_id"]:
        return None

    result["author"] = tds[3].text.strip()
    time_text = tds[4].text.strip()
    result["time_text"] = time_text
    result["post_time"] = parse_relative_time(time_text)

    result["content"] = ""
    result["source"] = "eastmoney"
    return result


def fetch_posts_scrapling(
    ts_code: str,
    start_time: datetime,
    max_pages: int = 3,
    page_delay: float = 2.0,
) -> List[Dict]:
    """用 Scrapling DynamicFetcher 抓取东方财富帖子（分页）。

    Args:
        ts_code: 如 SH688615
        start_time: 开始时间
        max_pages: 最大翻页数
        page_delay: 翻页间隔秒数

    Returns:
        帖子 dict 列表
    """
    from scrapling.fetchers import DynamicFetcher

    all_posts: List[Dict] = []
    code = _extract_code(ts_code)

    t0 = time.time()

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            url = f"https://guba.eastmoney.com/list,{code}.html"
        else:
            url = f"https://guba.eastmoney.com/list,{code}_{page_num}.html"

        logger.info("Fetching page %d: %s", page_num, url)
        page = DynamicFetcher.fetch(url, headless=True, network_idle=True, timeout=60000)

        trs = page.css("tr")
        rows = []
        for tr in trs:
            tds = tr.css("td")
            if len(tds) >= 5:
                text0 = tds[0].text.strip()
                if text0 and text0 != "阅读":
                    rows.append(tr)

        logger.info("Page %d: found %d rows", page_num, len(rows))

        if not rows:
            break

        batch_posts = []
        for row in rows:
            post = extract_post_scrapling(row)
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

        time.sleep(page_delay)

    logger.info("Total posts: %d, total time: %.1fs", len(all_posts), time.time() - t0)
    return all_posts


def compare_with_selenium():
    """用 Selenium 版抓同样数据做对比。"""
    from sentiment.eastmoney_scraper import fetch_posts as fetch_em_selenium

    start = datetime.now() - timedelta(days=7)
    logger.info("Running Selenium version for comparison...")

    t0 = time.time()
    sel_posts = fetch_em_selenium("SH688615", start, max_pages=3)
    sel_elapsed = time.time() - t0
    logger.info("Selenium: %d posts in %.1fs", len(sel_posts), sel_elapsed)

    return sel_posts, sel_elapsed


def main():
    print("=" * 60)
    print("东方财富 Scrapling vs Selenium 对比测试")
    print("=" * 60)

    ts_code = "SH688615"
    start = datetime.now() - timedelta(days=30)

    print("\n--- Scrapling DynamicFetcher ---")
    t0 = time.time()
    posts = fetch_posts_scrapling(ts_code, start, max_pages=3)
    sc_elapsed = time.time() - t0
    print(f"Scrapling: {len(posts)} posts in {sc_elapsed:.1f}s")

    if posts:
        print("\n前5条:")
        for p in posts[:5]:
            print(f"  {p['post_time']} | {p['author']:12s} | {p['title'][:60]}")

        print("\n字段验证:")
        p = posts[0]
        fields = ['ts_code','post_id','author','post_time','title','content','link','source']
        all_ok = all(k in p for k in fields)
        for k in fields:
            print(f"  {k}: {'OK' if k in p else 'MISSING'} = {str(p.get(k))[:60]}")
        print(f"  Fields complete: {all_ok}")

    # Selenium 对比
    print("\n--- Selenium 版对比 ---")
    sel_posts, sel_elapsed = compare_with_selenium()
    print(f"Selenium: {len(sel_posts)} posts in {sel_elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("东方财富平台结论:")
    print(f"  Scrapling: {len(posts)} posts, {sc_elapsed:.1f}s")
    print(f"  Selenium:  {len(sel_posts)} posts, {sel_elapsed:.1f}s")
    print("  东方财富无明显反爬，DynamicFetcher 即可工作")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())