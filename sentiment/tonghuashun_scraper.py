#!/usr/bin/env python3
"""
Purpose: 同花顺论股股票帖子抓取核心逻辑
Inputs:   股票代码 ts_code、开始时间 start_time
Outputs:  帖子列表（dict列表）
How to Run:
    本模块不直接运行，由 cli.py 调用
Examples:
    from sentiment.tonghuashun_scraper import fetch_posts
    posts = fetch_posts("SH688615", datetime(2026, 4, 1))
Side Effects: 启动 Selenium WebDriver 访问同花顺
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


def extract_post_from_item(item) -> Optional[Dict]:
    """从同花顺帖子列表的单项 DOM 提取帖子信息。"""
    result: Dict = {}

    # 标题与链接
    title_el = None
    try:
        title_el = item.find_element(By.CSS_SELECTOR, ".item-title")
    except Exception:
        try:
            title_el = item.find_element(By.CSS_SELECTOR, ".title")
        except Exception:
            pass

    if title_el:
        result["title"] = (title_el.get_attribute("textContent") or "").strip()
        href = title_el.get_attribute("href") or ""
        if href.startswith("http"):
            result["link"] = href
        else:
            result["link"] = f"https://basic.10jqka.com.cn{href}"
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
    result["author"] = ""
    try:
        author_el = item.find_element(By.CSS_SELECTOR, ".author-name")
        result["author"] = (author_el.get_attribute("textContent") or "").strip()
    except Exception:
        try:
            author_el = item.find_element(By.CSS_SELECTOR, ".author")
            result["author"] = (author_el.get_attribute("textContent") or "").strip()
        except Exception:
            pass

    # 时间
    time_text = ""
    try:
        time_el = item.find_element(By.CSS_SELECTOR, ".item-time")
        time_text = (time_el.get_attribute("textContent") or "").strip()
    except Exception:
        try:
            time_el = item.find_element(By.CSS_SELECTOR, ".time")
            time_text = (time_el.get_attribute("textContent") or "").strip()
        except Exception:
            pass
    result["time_text"] = time_text
    result["post_time"] = parse_relative_time(time_text)

    # 内容摘要
    result["content"] = ""
    try:
        content_el = item.find_element(By.CSS_SELECTOR, ".item-content")
        result["content"] = (content_el.get_attribute("textContent") or "").strip()
    except Exception:
        try:
            content_el = item.find_element(By.CSS_SELECTOR, ".content")
            result["content"] = (content_el.get_attribute("textContent") or "").strip()
        except Exception:
            pass

    result["source"] = "tonghuashun"
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
    code = _extract_code(ts_code)

    try:
        url = f"https://stockpage.10jqka.com.cn/{code}/"
        logger.info("Opening %s", url)
        driver.get(url)
        time.sleep(5)

        # 尝试点击"股吧"或"讨论"标签
        try:
            tabs = driver.find_elements(By.XPATH, "//*[contains(text(),'股吧') or contains(text(),'讨论') or contains(text(),'社区')]")
            for tab in tabs:
                if tab.tag_name in ["a", "button", "span", "div"]:
                    tab.click()
                    time.sleep(3)
                    break
        except Exception:
            pass

        for scroll_idx in range(max_scrolls):
            items = driver.find_elements(By.CSS_SELECTOR, ".list-item")
            if not items:
                items = driver.find_elements(By.CSS_SELECTOR, ".item")
            if not items:
                items = driver.find_elements(By.CSS_SELECTOR, ".post-item")

            logger.info("Scroll %d: found %d items", scroll_idx, len(items))

            if not items:
                break

            batch_posts = []
            for item in items:
                post = extract_post_from_item(item)
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
    posts = fetch_posts("SH688615", start, max_scrolls=3)
    print(f"Fetched {len(posts)} posts")
    for p in posts[:5]:
        print(f"{p['post_time']} | {p['author']} | {p['title'][:60]}")
