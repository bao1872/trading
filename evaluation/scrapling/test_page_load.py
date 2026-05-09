#!/usr/bin/env python3
"""
Purpose: Scrapling 三平台页面加载可行性测试
Inputs:   三平台 URL（雪球/东方财富/同花顺）
Outputs:  终端打印 PASS/FAIL 结果，含加载时间
How to Run:
    python evaluation/scrapling/test_page_load.py
Examples:
    python evaluation/scrapling/test_page_load.py
Side Effects: 启动 Playwright 浏览器（headless），访问三平台网站，不写入数据库
"""

import time
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_STOCK = "SH688615"
TEST_CODE = "688615"

TESTS = [
    {
        "name": "雪球",
        "url": f"https://xueqiu.com/S/{TEST_STOCK}",
        "expected_title_keywords": ["雪球", TEST_CODE],
    },
    {
        "name": "东方财富",
        "url": f"https://guba.eastmoney.com/list,{TEST_CODE}.html",
        "expected_title_keywords": ["东方财富", TEST_CODE],
    },
    {
        "name": "同花顺",
        "url": f"https://stockpage.10jqka.com.cn/{TEST_CODE}/",
        "expected_title_keywords": ["10jqka", "同花顺", TEST_CODE],
    },
]


def test_with_dynamic_fetcher(test_info):
    """用 DynamicFetcher（基础浏览器模式）测试。"""
    from scrapling.fetchers import DynamicFetcher

    t0 = time.time()
    try:
        page = DynamicFetcher.fetch(
            test_info["url"],
            headless=True,
            network_idle=True,
            timeout=30000,
        )
        elapsed = time.time() - t0
        title = page.css("title::text").get() or ""
        status = page.status

        has_keywords = any(kw in title for kw in test_info["expected_title_keywords"])

        logger.info(
            "[DynamicFetcher] %s: status=%s, title=%.80s, time=%.1fs, key_match=%s",
            test_info["name"], status, title, elapsed, has_keywords,
        )

        if status in (200, 0) and has_keywords:
            return True, elapsed, "DynamicFetcher OK"
        elif status == 403:
            return False, elapsed, f"403 Forbidden (title: {title})"
        elif not has_keywords:
            return False, elapsed, f"Title mismatch, got: {title[:100]}"
        else:
            return False, elapsed, f"status={status}, title={title[:100]}"
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("[DynamicFetcher] %s: ERROR %s", test_info["name"], e)
        return False, elapsed, str(e)[:200]


def test_with_stealthy_fetcher(test_info):
    """用 StealthyFetcher（反检测模式）测试。"""
    from scrapling.fetchers import StealthyFetcher

    t0 = time.time()
    try:
        page = StealthyFetcher.fetch(
            test_info["url"],
            headless=True,
            network_idle=True,
            timeout=30000,
        )
        elapsed = time.time() - t0
        title = page.css("title::text").get() or ""
        status = page.status

        has_keywords = any(kw in title for kw in test_info["expected_title_keywords"])

        logger.info(
            "[StealthyFetcher] %s: status=%s, title=%.80s, time=%.1fs, key_match=%s",
            test_info["name"], status, title, elapsed, has_keywords,
        )

        if status in (200, 0) and has_keywords:
            return True, elapsed, "StealthyFetcher OK"
        elif status == 403:
            return False, elapsed, f"403 Forbidden (title: {title})"
        elif not has_keywords:
            return False, elapsed, f"Title mismatch, got: {title[:100]}"
        else:
            return False, elapsed, f"status={status}, title={title[:100]}"
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("[StealthyFetcher] %s: ERROR %s", test_info["name"], e)
        return False, elapsed, str(e)[:200]


def main():
    print("=" * 60)
    print("Scrapling 三平台页面加载可行性测试")
    print("=" * 60)

    results = {}

    for fetcher_name, fetcher_fn in [
        ("DynamicFetcher", test_with_dynamic_fetcher),
        ("StealthyFetcher", test_with_stealthy_fetcher),
    ]:
        print(f"\n--- {fetcher_name} 测试 ---")
        for ti in TESTS:
            ok, elapsed, msg = fetcher_fn(ti)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {ti['name']:10s} | {elapsed:6.2f}s | {msg[:120]}")
            results[(fetcher_name, ti["name"])] = (ok, elapsed, msg)

    print("\n" + "=" * 60)
    print("汇总:")
    print("=" * 60)
    for (fn, name), (ok, elapsed, msg) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fn:20s} | {name:10s} | {elapsed:.2f}s")

    all_pass = all(ok for (ok, _, _) in results.values())
    print(f"\n总体结果: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())