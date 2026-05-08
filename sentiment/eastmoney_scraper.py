#!/usr/bin/env python3
"""
Purpose: 东方财富股吧股票帖子抓取核心逻辑
Inputs:   股票代码 ts_code、开始时间 start_time
Outputs:  帖子列表（dict列表）
How to Run:
    本模块不直接运行，由 cli.py 调用
Examples:
    from sentiment.eastmoney_scraper import fetch_posts
    posts = fetch_posts("SH688615", datetime(2026, 4, 1))
Side Effects: 启动 Selenium WebDriver 访问东方财富
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

from selenium.webdriver.common.by import By

from sentiment.scraper_utils import create_driver, parse_relative_time

logger = logging.getLogger(__name__)


def _extract_code(ts_code: str) -> str:
    """从 ts_code 提取纯数字代码。"""
    return "".join(filter(str.isdigit, ts_code))


def extract_post_from_row(row) -> Optional[Dict]:
    """从东方财富帖子列表的单行 DOM 提取帖子信息。"""
    result: Dict = {}

    tds = row.find_elements(By.TAG_NAME, "td")
    if len(tds) < 5:
        return None

    # 阅读数
    result["read_count"] = (tds[0].get_attribute("textContent") or "").strip()
    # 评论数
    result["comment_count"] = (tds[1].get_attribute("textContent") or "").strip()

    # 标题与链接
    title_el = tds[2]
    a_tag = None
    try:
        a_tag = title_el.find_element(By.TAG_NAME, "a")
    except Exception:
        pass

    if a_tag:
        result["title"] = (a_tag.get_attribute("textContent") or "").strip()
        href = a_tag.get_attribute("href") or ""
        if href.startswith("http"):
            result["link"] = href
        else:
            result["link"] = f"https://guba.eastmoney.com{href}"
    else:
        result["title"] = (title_el.get_attribute("textContent") or "").strip()
        result["link"] = ""

    # post_id 从链接中提取
    if result["link"]:
        parts = result["link"].rstrip("/").split(",")
        result["post_id"] = parts[-1].replace(".html", "") if parts else ""
    else:
        result["post_id"] = ""

    if not result["post_id"]:
        return None

    # 作者
    result["author"] = (tds[3].get_attribute("textContent") or "").strip()

    # 时间
    time_text = (tds[4].get_attribute("textContent") or "").strip()
    result["time_text"] = time_text
    result["post_time"] = parse_relative_time(time_text)

    # 内容：东方财富列表页通常只有标题，正文需进详情页（暂不抓取）
    result["content"] = ""
    result["source"] = "eastmoney"

    return result


def fetch_posts(
    ts_code: str,
    start_time: datetime,
    max_pages: int = 10,
    page_delay: float = 2.0,
) -> List[Dict]:
    """抓取某股票从 start_time 到现在的所有帖子。

    Args:
        ts_code: 股票代码，如 "SH688615"
        start_time: 只抓取发布时间 >= 该时间的帖子
        max_pages: 最大翻页数，防止死循环
        page_delay: 每次翻页后等待秒数

    Returns:
        帖子字典列表，每个字典包含：
        ts_code, post_id, author, post_time, title, content, link, time_text, source
    """
    driver = create_driver()
    all_posts: List[Dict] = []
    code = _extract_code(ts_code)

    try:
        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"https://guba.eastmoney.com/list,{code}.html"
            else:
                url = f"https://guba.eastmoney.com/list,{code}_{page}.html"

            logger.info("Opening %s", url)
            driver.get(url)
            time.sleep(3)

            # 东方财富帖子列表以 table 形式展示，每行是一个 tr
            trs = driver.find_elements(By.CSS_SELECTOR, "tr")
            rows = []
            for tr in trs:
                tds = tr.find_elements(By.TAG_NAME, "td")
                if len(tds) >= 5:
                    text0 = (tds[0].get_attribute("textContent") or "").strip()
                    if text0 and text0 != "阅读":
                        rows.append(tr)

            logger.info("Page %d: found %d rows", page, len(rows))

            if not rows:
                break

            batch_posts = []
            for row in rows:
                post = extract_post_from_row(row)
                if post is None:
                    continue
                post["ts_code"] = ts_code
                batch_posts.append(post)

            if not batch_posts:
                break

            batch_posts.sort(key=lambda x: x["post_time"], reverse=True)
            oldest_in_batch = batch_posts[-1]["post_time"]
            logger.info(
                "Batch range: %s ~ %s",
                batch_posts[0]["post_time"],
                oldest_in_batch,
            )

            seen_ids = {p["post_id"] for p in all_posts}
            for post in batch_posts:
                if post["post_id"] not in seen_ids and post["post_time"] >= start_time:
                    all_posts.append(post)

            if oldest_in_batch < start_time:
                logger.info("Oldest post %s < start_time %s, stopping", oldest_in_batch, start_time)
                break

            # 点击下一页
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, ".pager .next")
                if "disabled" in (next_btn.get_attribute("class") or ""):
                    logger.info("No more pages")
                    break
                next_btn.click()
                time.sleep(page_delay)
            except Exception:
                logger.info("No next page button found")
                break

        logger.info("Total posts fetched: %d", len(all_posts))
        return all_posts

    finally:
        driver.quit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.now() - timedelta(days=7)
    posts = fetch_posts("SH688615", start, max_pages=2)
    print(f"Fetched {len(posts)} posts")
    for p in posts[:5]:
        print(f"{p['post_time']} | {p['author']} | {p['title'][:60]}")
