#!/usr/bin/env python3
"""
Purpose: 雪球网股票帖子抓取核心逻辑
Inputs:   股票代码 ts_code、开始时间 start_time
Outputs:  帖子列表（dict列表）
How to Run:
    本模块不直接运行，由 cli.py 或 db_operations.py 调用
Examples:
    from sentiment.xueqiu_scraper import fetch_posts
    posts = fetch_posts("SH688615", datetime(2026, 4, 1))
Side Effects: 启动 Selenium WebDriver 访问雪球网
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from selenium.webdriver.common.by import By

from sentiment.scraper_utils import create_driver, parse_relative_time

logger = logging.getLogger(__name__)


def extract_post_from_article(article) -> Optional[Dict]:
    """从单条 article DOM 提取帖子信息，提取失败返回 None。"""
    result: Dict = {}

    # 作者
    try:
        author_el = article.find_element(By.CSS_SELECTOR, ".user-name")
        result["author"] = author_el.get_attribute("textContent").strip()
    except Exception:
        result["author"] = ""

    # 时间（去掉 "· 来自..."）
    try:
        time_el = article.find_element(By.CSS_SELECTOR, ".date-and-source")
        time_text = time_el.get_attribute("textContent").split("·")[0].strip()
        result["time_text"] = time_text
        result["post_time"] = parse_relative_time(time_text)
    except Exception:
        return None

    # 链接
    try:
        link_el = article.find_element(By.CSS_SELECTOR, "a[data-id]")
        data_id = link_el.get_attribute("data-id")
        result["post_id"] = data_id
        result["link"] = f"https://xueqiu.com/{data_id}"
    except Exception:
        try:
            link_el = article.find_element(By.CSS_SELECTOR, ".date-and-source")
            href = link_el.get_attribute("href")
            if href:
                result["link"] = href
                parts = href.rstrip("/").split("/")
                result["post_id"] = parts[-1] if parts else ""
        except Exception:
            return None

    if not result.get("post_id"):
        return None

    # 内容
    try:
        content_el = article.find_element(By.CSS_SELECTOR, ".content--description")
        full_text = content_el.get_attribute("textContent").strip()
    except Exception:
        try:
            content_el = article.find_element(By.CSS_SELECTOR, ".timeline__item__content")
            full_text = content_el.get_attribute("textContent").strip()
        except Exception:
            full_text = ""

    if full_text:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        result["title"] = lines[0] if lines else ""
        result["content"] = "\n".join(lines[1:]) if len(lines) > 1 else ""
    else:
        result["title"] = ""
        result["content"] = ""

    result["source"] = "xueqiu"
    return result


def fetch_posts(
    ts_code: str,
    start_time: datetime,
    max_scrolls: int = 10,
    scroll_delay: float = 2.0,
) -> List[Dict]:
    """抓取某股票从 start_time 到现在的所有帖子。

    Args:
        ts_code: 股票代码，如 "SH688615"
        start_time: 只抓取发布时间 >= 该时间的帖子
        max_scrolls: 最大滚动次数，防止死循环
        scroll_delay: 每次滚动后等待秒数

    Returns:
        帖子字典列表，每个字典包含：
        ts_code, post_id, author, post_time, title, content, link, time_text, source
    """
    driver = create_driver()
    all_posts: List[Dict] = []

    try:
        url = f"https://xueqiu.com/S/{ts_code}"
        logger.info("Opening %s", url)
        driver.get(url)
        time.sleep(5)

        for scroll_idx in range(max_scrolls):
            try:
                articles = driver.find_elements(By.TAG_NAME, "article")
            except Exception as e:
                logger.warning("Find elements timeout on scroll %d: %s", scroll_idx, e)
                break
            logger.info("Scroll %d: found %d articles", scroll_idx, len(articles))

            if not articles:
                break

            batch_posts = []
            for article in articles:
                post = extract_post_from_article(article)
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

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_delay)

        logger.info("Total posts fetched: %d", len(all_posts))
        return all_posts

    finally:
        driver.quit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.now() - timedelta(days=7)
    posts = fetch_posts("SH688615", start)
    print(f"Fetched {len(posts)} posts")
    for p in posts[:5]:
        print(f"{p['post_time']} | {p['author']} | {p['title'][:60]}")
