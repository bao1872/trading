#!/usr/bin/env python3
"""
Purpose: 舆情抓取公共工具（SSOT）：Selenium 驱动管理、时间解析、安全 DOM 提取
Inputs:   无
Outputs:  create_driver, parse_relative_time, safe_find_text 等公共函数
How to Run:
    本模块不直接运行，由各平台 scraper 导入使用
Examples:
    from sentiment.scraper_utils import create_driver, parse_relative_time
    driver = create_driver()
    dt = parse_relative_time("05-03 19:11")
Side Effects: 启动 Selenium WebDriver
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By

logger = logging.getLogger(__name__)


def create_driver() -> webdriver.Chrome:
    """创建并返回配置好反检测的 Chrome WebDriver。"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-extensions")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    service = ChromeService("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        },
    )
    return driver


def parse_relative_time(text: str) -> datetime:
    """将中文相对时间转为绝对 datetime（假设为北京时间当年）。"""
    now = datetime.now()
    text = text.strip()

    # 格式如 "05-03 19:11" 或 "修改于05-03 19:11"
    m = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", text)
    if m:
        month, day, hour, minute = map(int, m.groups())
        dt = datetime(now.year, month, day, hour, minute)
        if dt > now + timedelta(days=1):
            dt = dt.replace(year=now.year - 1)
        return dt

    # 格式如 "昨天 08:14"
    m = re.match(r"昨天\s+(\d{2}):(\d{2})", text)
    if m:
        hour, minute = map(int, m.groups())
        dt = (now - timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        return dt

    # 格式如 "今天 10:30"
    m = re.match(r"今天\s+(\d{2}):(\d{2})", text)
    if m:
        hour, minute = map(int, m.groups())
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt

    # 纯时间如 "18:12" 视为今天
    m = re.match(r"(\d{2}):(\d{2})", text)
    if m:
        hour, minute = map(int, m.groups())
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt

    # 格式如 "2026-05-03 19:11"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", text)
    if m:
        year, month, day, hour, minute = map(int, m.groups())
        return datetime(year, month, day, hour, minute)

    # 格式如 "刚刚"、"1分钟前"、"1小时前"
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


def safe_find_text(element, selector: str, by: By = By.CSS_SELECTOR) -> str:
    """安全提取 DOM 元素的 textContent，失败返回空字符串。"""
    try:
        el = element.find_element(by, selector)
        return el.get_attribute("textContent") or ""
    except Exception:
        return ""


def safe_find_attr(element, selector: str, attr: str, by: By = By.CSS_SELECTOR) -> str:
    """安全提取 DOM 元素的属性值，失败返回空字符串。"""
    try:
        el = element.find_element(by, selector)
        return el.get_attribute(attr) or ""
    except Exception:
        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # 自测时间解析
    test_cases = [
        "05-03 19:11",
        "昨天 08:14",
        "今天 10:30",
        "18:12",
        "刚刚",
        "5分钟前",
        "2小时前",
        "2026-05-03 19:11",
    ]
    for tc in test_cases:
        print(f"{tc:20s} -> {parse_relative_time(tc)}")
