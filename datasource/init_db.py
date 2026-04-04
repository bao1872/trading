# -*- coding: utf-8 -*-
"""
数据库初始化脚本 - 检查远程 PostgreSQL 数据库表结构

Purpose:
    检查远程 PostgreSQL 数据库的表结构是否完整

Inputs:
    无（从 config.py 读取数据库连接）

Outputs:
    表结构检查报告

How to Run:
    python datasource/init_db.py

Examples:
    python datasource/init_db.py
    python datasource/init_db.py --create-missing

Side Effects:
    无（只读操作，不会修改数据库结构）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATABASE_URL
from datasource.database import get_session, table_exists
from sqlalchemy import text


def check_tables():
    """检查数据库表是否存在"""
    expected_tables = [
        'stock_k_data',
        'first_day_launch_scores',
        'financial_quarterly_data',
        'stock_financial_score_pool',
        'stock_top10_holders_tushare',
        'stock_anomaly_signals',
        'concept_signals',
        'theme_signals',
        'breakout_dir_turn_events',
        'breakout_pullback_buy_events',
        'limit_up_signals',
        'stock_watchlist',
        'saved_filters'
    ]

    print("=" * 60)
    print("远程 PostgreSQL 数据库表结构检查")
    print("=" * 60)
    print(f"数据库 URL: {DATABASE_URL}")
    print("=" * 60)

    with get_session() as session:
        # 测试连接
        result = session.execute(text("SELECT version()"))
        pg_version = result.fetchone()[0]
        print(f"✅ 连接成功")
        print(f"PostgreSQL 版本: {pg_version}")
        print("=" * 60)

        # 检查表
        print(f"\n共 {len(expected_tables)} 张核心表待检查:\n")
        existing_count = 0
        for table_name in expected_tables:
            exists = table_exists(session, table_name)
            status = "✅ 存在" if exists else "❌ 不存在"
            print(f"  {status} {table_name}")
            if exists:
                existing_count += 1

        print("=" * 60)
        print(f"检查结果: {existing_count}/{len(expected_tables)} 张表存在")
        print("=" * 60)

        # 列出所有表
        result = session.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        all_tables = [row[0] for row in result.fetchall()]
        print(f"\n数据库中共有 {len(all_tables)} 张表:")
        for t in all_tables:
            print(f"  - {t}")

    return existing_count == len(expected_tables)


def init_tables():
    """
    初始化数据库表结构检查

    在 PostgreSQL 环境下，主要检查核心表是否存在
    如果表不存在，会打印警告信息

    Returns:
        bool: 所有核心表是否都存在
    """
    print("=" * 60)
    print("初始化数据库表结构检查")
    print("=" * 60)

    all_exist = check_tables()

    if not all_exist:
        print("\n⚠️  警告：部分核心表不存在")
        print("   请确保 PostgreSQL 数据库中已创建所有必要的表")
        print("   可以联系数据库管理员或使用 pgloader 迁移数据")
    else:
        print("\n✅ 数据库表结构检查通过")

    return all_exist


def main():
    import argparse
    parser = argparse.ArgumentParser(description="检查远程 PostgreSQL 数据库表结构")
    parser.add_argument("--create-missing", action="store_true",
                        help="创建缺失的表（需手动在 PostgreSQL 中执行 DDL）")
    args = parser.parse_args()

    all_exist = check_tables()

    if not all_exist:
        print("\n⚠️  部分表不存在，请先在 PostgreSQL 中创建表结构")
        print("   可以使用 pgloader 或手动执行 DDL 迁移数据")
    else:
        print("\n✅ 所有核心表已存在")


if __name__ == "__main__":
    main()
