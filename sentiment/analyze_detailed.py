#!/usr/bin/env python3
"""
Purpose: 深入分析地平线机器人雪球评论，输出详细博主画像
Inputs:   无
Outputs:  详细博主分析报告（控制台+JSON）
How to Run:
    cd /root/trading && PYTHONPATH=/root/trading python sentiment/analyze_detailed.py
Examples:
    python sentiment/analyze_detailed.py
Side Effects: 启动 Playwright headless 浏览器访问雪球，生成JSON报告
"""

import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from sentiment.xueqiu_scraper import fetch_posts

PROFESSIONAL_KEYWORDS = [
    "营收", "利润", "毛利率", "PE", "估值", "财报", "业绩", "收入", "亏损",
    "芯片", "算法", "NOA", "智驾", "ADAS", "量产", "算力", "传感器",
    "渗透率", "市场份额", "订单", "定点", "交付", "产能", "竞品",
    "买入", "卖出", "加仓", "减仓", "持仓", "资金流", "南向", "北向",
    "回购", "配股", "空头", "做空", "市值", "港股", "IPO",
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    ts_code = "09660"
    start_time = datetime.now() - timedelta(days=30)
    
    print("抓取地平线机器人评论...")
    posts = fetch_posts(ts_code, start_time, max_scrolls=15, scroll_delay=2.0)
    print(f"获取到 {len(posts)} 条帖子\n")
    
    author_data = defaultdict(lambda: {"posts": [], "keywords": [], "total_length": 0})
    
    for p in posts:
        author = p["author"]
        content = p.get("content", "")
        author_data[author]["posts"].append(p)
        author_data[author]["total_length"] += len(content)
        
        found_keywords = [kw for kw in PROFESSIONAL_KEYWORDS if kw in content]
        author_data[author]["keywords"].extend(found_keywords)
    
    print("=" * 100)
    print("地平线机器人(09660) 雪球博主质量分析")
    print("=" * 100)
    
    ranked_authors = []
    for author, data in author_data.items():
        post_count = len(data["posts"])
        avg_len = data["total_length"] / post_count
        unique_keywords = list(set(data["keywords"]))
        keyword_count = len(unique_keywords)
        
        quality = "普通"
        if avg_len > 100 and keyword_count >= 2:
            quality = "高质量"
        elif avg_len > 80 or keyword_count >= 1:
            quality = "中等"
        
        ranked_authors.append({
            "author": author,
            "quality": quality,
            "post_count": post_count,
            "avg_length": round(avg_len),
            "keywords": unique_keywords,
            "keyword_count": keyword_count,
            "posts": data["posts"],
        })
    
    ranked_authors.sort(key=lambda x: (x["keyword_count"], x["avg_length"]), reverse=True)
    
    for i, a in enumerate(ranked_authors, 1):
        print(f"\n{'─' * 100}")
        print(f"#{i} 【{a['author']}】 | 质量等级: {a['quality']}")
        print(f"   发帖数: {a['post_count']} | 平均字数: {a['avg_length']} | 专业词数: {a['keyword_count']}")
        if a['keywords']:
            print(f"   涉及关键词: {', '.join(a['keywords'])}")
        
        for j, p in enumerate(a["posts"], 1):
            print(f"\n   帖子 {j} ({p.get('post_time', 'N/A').strftime('%Y-%m-%d %H:%M')})")
            print(f"   链接: {p.get('link', 'N/A')}")
            print(f"   内容: {p.get('content', '')[:200]}")
    
    print(f"\n{'=' * 100}")
    print("总结")
    print(f"{'=' * 100}")
    high_quality = [a for a in ranked_authors if a['quality'] == '高质量']
    medium = [a for a in ranked_authors if a['quality'] == '中等']
    print(f"高质量博主: {len(high_quality)} 位")
    for a in high_quality:
        print(f"  - {a['author']} ({a['post_count']}帖, {a['avg_length']}字/帖, {a['keyword_count']}个专业词)")
    
    print(f"\n中等质量博主: {len(medium)} 位")
    for a in medium:
        print(f"  - {a['author']} ({a['post_count']}帖, {a['avg_length']}字/帖)")
    
    output = {
        "analysis_date": datetime.now().isoformat(),
        "stock_code": ts_code,
        "total_posts": len(posts),
        "total_authors": len(author_data),
        "ranked_authors": [
            {
                "author": a["author"],
                "quality": a["quality"],
                "post_count": a["post_count"],
                "avg_length": a["avg_length"],
                "keywords": a["keywords"],
            }
            for a in ranked_authors
        ]
    }
    
    os.makedirs("sentiment/output", exist_ok=True)
    output_file = f"sentiment/output/xueqiu_detailed_{ts_code}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {output_file}")
