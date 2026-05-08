#!/usr/bin/env python3
"""
Purpose: 高质量博主发现与帖子时间线展示的命令行入口
Inputs:   命令行参数
Outputs:  终端打印博主排名或帖子时间线
How to Run:
    python -m sentiment.blogger_cli recommend --ts-code SH688615 --months 3 --top-n 10
    python -m sentiment.blogger_cli timeline --ts-code SH688615 --author "博主名" --months 3
Examples:
    # 查看合合信息近3个月高质量博主 Top 10
    python -m sentiment.blogger_cli recommend --ts-code SH688615 --months 3 --top-n 10

    # 仅看雪球平台
    python -m sentiment.blogger_cli recommend --ts-code SH688615 --months 3 --sources xueqiu

    # 展示某博主在合合信息下的时间线
    python -m sentiment.blogger_cli timeline --ts-code SH688615 --author "某博主名" --months 3
Side Effects: 读数据库 stock_sentiment_posts 表，不写入
"""

import argparse
import logging
import sys

from sentiment.blogger_analyzer import (
    analyze_bloggers,
    show_blogger_timeline,
    format_blogger_ranking,
    format_timeline,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="高质量评论博主发现与帖子时间线展示"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    recommend_parser = subparsers.add_parser("recommend", help="推荐高质量博主")
    recommend_parser.add_argument(
        "--ts-code", required=True, help="股票代码，如 SH688615"
    )
    recommend_parser.add_argument(
        "--months", type=int, default=3, help="时间窗口月数，默认3"
    )
    recommend_parser.add_argument(
        "--top-n", type=int, default=10, help="显示前N名，默认10"
    )
    recommend_parser.add_argument(
        "--sources",
        type=str,
        default="xueqiu,eastmoney",
        help="来源平台，逗号分隔，如 xueqiu,eastmoney，默认全部",
    )
    recommend_parser.add_argument(
        "--w-freq", type=float, default=0.6, help="频率权重，默认0.6"
    )
    recommend_parser.add_argument(
        "--w-content", type=float, default=0.4, help="内容深度权重，默认0.4"
    )

    timeline_parser = subparsers.add_parser("timeline", help="展示博主帖子时间线")
    timeline_parser.add_argument(
        "--ts-code", required=True, help="股票代码，如 SH688615"
    )
    timeline_parser.add_argument(
        "--author", required=True, help="博主用户名"
    )
    timeline_parser.add_argument(
        "--months", type=int, default=3, help="时间窗口月数，默认3"
    )
    timeline_parser.add_argument(
        "--sources",
        type=str,
        default="xueqiu,eastmoney",
        help="来源平台，逗号分隔，如 xueqiu,eastmoney，默认全部",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.command == "recommend":
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        rankings = analyze_bloggers(
            args.ts_code,
            months=args.months,
            sources=sources,
            top_n=args.top_n,
            w_freq=args.w_freq,
            w_content=args.w_content,
        )
        print(format_blogger_ranking(rankings))
        return 0

    elif args.command == "timeline":
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        timeline = show_blogger_timeline(
            args.ts_code,
            args.author,
            months=args.months,
            sources=sources,
        )
        print(format_timeline(timeline, args.author))
        return 0

    else:
        print("请指定子命令: recommend 或 timeline", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
