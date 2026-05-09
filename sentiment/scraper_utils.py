#!/usr/bin/env python3
"""
Purpose: 舆情抓取公共工具（SSOT）：中文相对时间解析
Inputs:   无
Outputs:  parse_relative_time 公共函数
How to Run:
    本模块不直接运行，由各平台 scraper 导入使用
Examples:
    from sentiment.scraper_utils import parse_relative_time
    dt = parse_relative_time("05-03 19:11")
Side Effects: 无
"""

import re
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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