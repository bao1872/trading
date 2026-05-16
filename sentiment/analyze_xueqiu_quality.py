#!/usr/bin/env python3
"""
Purpose: 分析雪球评论，识别高质量博主
Inputs:   ts_code (股票代码), days (时间窗口天数), max_scrolls (滚动次数)
Outputs:  高质量博主排名（控制台打印 + CSV 保存）
How to Run:
    cd /root/trading && PYTHONPATH=/root/trading python sentiment/analyze_xueqiu_quality.py --code 09660 --days 30 --scrolls 15
Examples:
    python sentiment/analyze_xueqiu_quality.py --code 09660 --days 30
    python sentiment/analyze_xueqiu_quality.py --code SH688615 --days 7 --scrolls 5
Side Effects: 启动 Playwright headless 浏览器访问雪球，生成 CSV 文件到 sentiment/output/
"""

import os
import re
import sys
import argparse
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict

import pandas as pd

from sentiment.xueqiu_scraper import fetch_posts

logger = logging.getLogger(__name__)

PROFESSIONAL_KEYWORDS = {
    "财务类": ["营收", "利润", "毛利率", "净利率", "PE", "PB", "估值", "财报", "业绩",
              "收入", "亏损", "每股收益", "ROE", "现金流", "负债", "资产", "分红",
              "净利润", "扣非", "同比", "环比", "增速", "预期"],
    "技术类": ["芯片", "算法", "NOA", "智驾", "ADAS", "量产", "算力", "传感器",
              "激光雷达", "摄像头", "感知", "决策", "规划", "控制", "域控制器",
              "SoC", "FPGA", "GPU", "CPU", "NPU", "大模型", "端到端", "BEV"],
    "行业类": ["渗透率", "市场份额", "订单", "定点", "交付", "产能", "竞品",
              "特斯拉", "华为", "比亚迪", "理想", "小鹏", "蔚来", "配套",
              "供应链", "客户", "合作", "战略", "投资", "并购"],
    "交易类": ["买入", "卖出", "加仓", "减仓", "持仓", "止盈", "止损", "支撑",
              "压力", "突破", "反弹", "回调", "资金流", "主力", "北向", "南向"],
}

CONTENT_LENGTH_SCORES = [(50, 1), (200, 2), (500, 3), (float("inf"), 4)]

def score_content_length(length: int) -> int:
    for threshold, score in CONTENT_LENGTH_SCORES:
        if length <= threshold:
            return score
    return 4


def count_professional_keywords(text: str) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for keywords in PROFESSIONAL_KEYWORDS.values():
        for kw in keywords:
            if kw.lower() in text_lower:
                count += 1
    return count


def has_independent_title(title: str, content: str) -> bool:
    if not title or not content:
        return False
    if title.startswith("回复"):
        return False
    if title.startswith("$") and title.count("$") == 2:
        return False
    return len(title) > 5


def is_recent(post_time: datetime, days: int = 7) -> bool:
    return (datetime.now() - post_time).days <= days


def analyze_quality(posts: List[Dict]) -> pd.DataFrame:
    author_posts = defaultdict(list)
    for p in posts:
        author_posts[p["author"]].append(p)
    
    results = []
    for author, posts_list in author_posts.items():
        post_count = len(posts_list)
        total_length = sum(len(p.get("content", "")) for p in posts_list)
        avg_length = total_length / post_count if post_count else 0
        
        all_content = " ".join(p.get("content", "") for p in posts_list)
        keyword_count = count_professional_keywords(all_content)
        
        independent_titles = sum(1 for p in posts_list if has_independent_title(p.get("title", ""), p.get("content", "")))
        independent_ratio = independent_titles / post_count if post_count else 0
        
        recent_count = sum(1 for p in posts_list if is_recent(p.get("post_time", datetime.min)))
        recent_ratio = recent_count / post_count if post_count else 0
        
        length_score = score_content_length(avg_length)
        keyword_score = min(int(keyword_count / 2), 4)
        frequency_score = min(post_count, 4)
        originality_score = int(independent_ratio * 4)
        recency_score = int(recent_ratio * 4)
        
        total_score = (
            length_score * 0.30 +
            keyword_score * 0.25 +
            frequency_score * 0.20 +
            originality_score * 0.15 +
            recency_score * 0.10
        ) * 25
        
        top_posts = sorted(posts_list, key=lambda x: len(x.get("content", "")), reverse=True)[:3]
        top_titles = " | ".join(p.get("title", "")[:40] for p in top_posts)
        
        results.append({
            "博主": author,
            "总评分": round(total_score, 1),
            "发帖数": post_count,
            "平均内容长度": round(avg_length),
            "专业关键词数": keyword_count,
            "原创标题占比": round(independent_ratio * 100, 1),
            "近7天活跃度": round(recent_ratio * 100, 1),
            "代表帖子": top_titles,
        })
    
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values("总评分", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="分析雪球评论，识别高质量博主")
    parser.add_argument("--code", default="09660", help="股票代码 (如 09660)")
    parser.add_argument("--days", type=int, default=30, help="时间窗口天数 (默认30)")
    parser.add_argument("--scrolls", type=int, default=15, help="页面滚动次数 (默认15)")
    parser.add_argument("--delay", type=float, default=2.0, help="滚动延迟秒数 (默认2.0)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    ts_code = args.code
    start_time = datetime.now() - timedelta(days=args.days)
    
    print(f"正在抓取 {ts_code} 最近 {args.days} 天的评论...")
    posts = fetch_posts(ts_code, start_time, max_scrolls=args.scrolls, scroll_delay=args.delay)
    print(f"获取到 {len(posts)} 条有效帖子\n")
    
    if not posts:
        print("未获取到数据，退出")
        return
    
    df = analyze_quality(posts)
    
    print("=" * 120)
    print("高质量博主排名")
    print("=" * 120)
    print(df[["博主", "总评分", "发帖数", "平均内容长度", "专业关键词数"]].head(20).to_string(index=False))
    print("=" * 120)
    
    print("\n详细分析:")
    for _, row in df.head(10).iterrows():
        print(f"\n【{row['博主']}】评分: {row['总评分']}")
        print(f"  发帖数: {row['发帖数']} | 平均长度: {row['平均内容长度']} | 专业词: {row['专业关键词数']}")
        print(f"  代表帖子: {row['代表帖子'][:100]}...")
    
    os.makedirs("sentiment/output", exist_ok=True)
    output_file = f"sentiment/output/xueqiu_quality_{ts_code}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
