#!/usr/bin/env python3
"""
Purpose: Scrapling 迁移后 5 只股票双平台验证测试
Inputs:   5 只股票列表
Outputs:  终端打印逐股结果 + 汇总表格 + PASS/FAIL
How to Run:
    PYTHONPATH=/root/trading python evaluation/scrapling/migration_test.py
Examples:
    PYTHONPATH=/root/trading python evaluation/scrapling/migration_test.py
Side Effects: 启动 Playwright 浏览器（headless），访问雪球和东方财富，不写入数据库
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STOCKS = [
    ("SH688615", "合合信息"),
    ("SH600519", "贵州茅台"),
    ("SZ000001", "平安银行"),
    ("SH601398", "工商银行"),
    ("SZ300750", "宁德时代"),
]

START = datetime.now() - timedelta(days=30)

REQUIRED_FIELDS = ["ts_code", "post_id", "author", "post_time", "title", "content", "link", "source"]


def test_xueqiu(ts_code: str, name: str) -> Dict:
    """测试雪球平台 Scrapling 版。"""
    from sentiment.xueqiu_scraper import fetch_posts

    t0 = time.time()
    try:
        posts = fetch_posts(ts_code, START, max_scrolls=3, scroll_delay=2.0)
        elapsed = time.time() - t0
        fields_ok = all(k in posts[0] for k in REQUIRED_FIELDS) if posts else False
        return {
            "stock": name,
            "ts_code": ts_code,
            "platform": "xueqiu",
            "posts": len(posts),
            "time_s": round(elapsed, 1),
            "fields_ok": fields_ok,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "stock": name,
            "ts_code": ts_code,
            "platform": "xueqiu",
            "posts": 0,
            "time_s": round(elapsed, 1),
            "fields_ok": False,
            "error": str(e)[:200],
        }


def test_eastmoney(ts_code: str, name: str) -> Dict:
    """测试东方财富平台 Scrapling 版。"""
    from sentiment.eastmoney_scraper import fetch_posts

    t0 = time.time()
    try:
        posts = fetch_posts(ts_code, START, max_pages=3, page_delay=0.5)
        elapsed = time.time() - t0
        fields_ok = all(k in posts[0] for k in REQUIRED_FIELDS) if posts else False
        return {
            "stock": name,
            "ts_code": ts_code,
            "platform": "eastmoney",
            "posts": len(posts),
            "time_s": round(elapsed, 1),
            "fields_ok": fields_ok,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "stock": name,
            "ts_code": ts_code,
            "platform": "eastmoney",
            "posts": 0,
            "time_s": round(elapsed, 1),
            "fields_ok": False,
            "error": str(e)[:200],
        }


def main():
    print("=" * 80)
    print("  Scrapling 迁移验证：5 只股票 × 2 平台")
    print(f"  时间窗口: {START.strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 80)

    all_results: List[Dict] = []
    overall_t0 = time.time()

    for i, (ts_code, name) in enumerate(STOCKS, 1):
        print(f"\n{'─' * 80}")
        print(f"  [{i}/{len(STOCKS)}] {ts_code} {name}")
        print(f"{'─' * 80}")

        r_xq = test_xueqiu(ts_code, name)
        all_results.append(r_xq)
        xq_status = "PASS" if r_xq["posts"] > 0 and r_xq["fields_ok"] else "FAIL"
        print(f"  雪球:      [{xq_status}] {r_xq['posts']:4d} posts in {r_xq['time_s']:6.1f}s" +
              (f" | ERROR: {r_xq['error'][:60]}" if r_xq["error"] else ""))

        r_em = test_eastmoney(ts_code, name)
        all_results.append(r_em)
        em_status = "PASS" if r_em["posts"] > 0 and r_em["fields_ok"] else "FAIL"
        print(f"  东方财富:  [{em_status}] {r_em['posts']:4d} posts in {r_em['time_s']:6.1f}s" +
              (f" | ERROR: {r_em['error'][:60]}" if r_em["error"] else ""))

    total_elapsed = time.time() - overall_t0

    # ============ 汇总报告 ============
    print("\n" + "=" * 80)
    print("                       测 试 报 告")
    print("=" * 80)

    print(f"""
┌─────────────────┬──────────┬──────────┬────────────┬────────────┬───────┐
│ 股票            │ 雪球帖子    │ 雪球耗时    │ 东方帖子    │ 东方耗时    │ 状态  │
├─────────────────┼──────────┼──────────┼────────────┼────────────┼───────┤""")

    for ts_code, name in STOCKS:
        xq = next((r for r in all_results if r["ts_code"] == ts_code and r["platform"] == "xueqiu"), None)
        em = next((r for r in all_results if r["ts_code"] == ts_code and r["platform"] == "eastmoney"), None)

        xq_posts = xq["posts"] if xq else "N/A"
        xq_time = f"{xq['time_s']:.0f}s" if xq else "N/A"
        em_posts = em["posts"] if em else "N/A"
        em_time = f"{em['time_s']:.0f}s" if em else "N/A"

        xq_ok = xq and xq["posts"] > 0 and xq["fields_ok"]
        em_ok = em and em["posts"] > 0 and em["fields_ok"]
        if xq_ok and em_ok:
            status = "ALL PASS"
        elif xq_ok or em_ok:
            status = "PARTIAL"
        else:
            status = "FAIL"

        print(f"│ {name:7s} {ts_code:10s} │ {str(xq_posts):>4s}      │ {xq_time:>7s}  │ {str(em_posts):>4s}       │ {em_time:>7s}  │ {status:7s} │")

    total_xq = sum(r["posts"] for r in all_results if r["platform"] == "xueqiu" and r["posts"] > 0)
    total_em = sum(r["posts"] for r in all_results if r["platform"] == "eastmoney" and r["posts"] > 0)
    total = total_xq + total_em

    print(f"""├─────────────────┼──────────┼──────────┼────────────┼────────────┼───────┤
│ 合计            │ {total_xq:4d}      │          │ {total_em:4d}       │          │       │
└─────────────────┴──────────┴──────────┴────────────┴────────────┴───────┘
总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)  |  总帖子: {total} 条
""")

    # ============ 异常汇总 ============
    errors = [r for r in all_results if r["error"]]
    if errors:
        print("异常列表:")
        for e in errors:
            print(f"  [{e['platform']}] {e['ts_code']} {e['stock']}: {e['error'][:100]}")

    # ============ 字段验证 ============
    print("\n字段验证:")
    all_ok = True
    for r in all_results:
        if r["posts"] > 0 and not r["fields_ok"]:
            print(f"  MISSING: {r['ts_code']} {r['platform']} — 缺少必需字段")
            all_ok = False
    if all_ok:
        print("  所有非空结果包含完整的 8 字段 ✓")

    # ============ 最终判定 ============
    xq_pass = all(r["posts"] > 0 and r["fields_ok"] for r in all_results if r["platform"] == "xueqiu")
    em_pass = all(r["posts"] > 0 and r["fields_ok"] for r in all_results if r["platform"] == "eastmoney")

    print(f"\n雪球平台:    {'ALL PASS ✓' if xq_pass else 'SOME FAIL ✗'}")
    print(f"东方财富:    {'ALL PASS ✓' if em_pass else 'SOME FAIL ✗'}")

    final = xq_pass and em_pass
    print(f"\n总体判定:    {'PASS ✓ — 迁移成功' if final else 'FAIL ✗ — 需排查'}")
    return 0 if final else 1


if __name__ == "__main__":
    sys.exit(main())