#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入 stock_concepts_cache.xlsx 到数据库

Purpose: 将股票概念缓存文件导入到数据库
Inputs: stock_concepts_cache.xlsx 文件
Outputs: stock_pools 表中的数据
How to Run:
    # 从项目根目录运行
    python app/import_stock_cache_to_db.py

    # 或使用模块方式运行
    python -m app.import_stock_cache_to_db
Side Effects: 向 stock_pools 表插入/更新数据
"""

import os
import sys
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datasource.database import get_session
from app.models import get_create_sql
from sqlalchemy import text


def create_table_if_not_exists():
    """创建表（如果不存在）"""
    with get_session() as session:
        sql = get_create_sql("stock_pools")
        session.execute(text(sql))
        session.commit()
        print("✅ 已确保 stock_pools 表存在")


def import_stock_cache_from_excel(excel_path: str):
    """从 Excel 导入股票概念缓存到数据库"""
    if not os.path.exists(excel_path):
        print(f"❌ 错误：文件 {excel_path} 不存在")
        return False

    df = pd.read_excel(excel_path)
    print(f"📊 从 {excel_path} 读取到 {len(df)} 条股票记录")
    print(f"📋 列名：{df.columns.tolist()}")

    required_cols = ['name', 'ts_code']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误：缺少必需的列：{missing_cols}")
        print(f"✅ 可用列：{df.columns.tolist()}")
        return False

    df = df[required_cols].copy()

    original_count = len(df)
    df.drop_duplicates(subset=['ts_code'], keep='first', inplace=True)
    if len(df) < original_count:
        print(f"⚠️  去重：从 {original_count} 条记录去重至 {len(df)} 条")

    print(f"📈 最终导入 {len(df)} 条不重复的股票记录")

    with get_session() as session:
        session.execute(text("TRUNCATE TABLE stock_pools RESTART IDENTITY"))
        print("🗑️  已清空现有缓存数据")

        records = df.to_dict(orient='records')
        for record in records:
            sql = """
                INSERT INTO stock_pools (name, ts_code)
                VALUES (:name, :ts_code)
            """
            session.execute(text(sql), record)

        session.commit()
        print(f"✅ 成功导入 {len(records)} 条股票缓存记录到数据库")

    return True


def main():
    """主函数"""
    excel_path = "stock_concepts_cache.xlsx"

    print("🚀 开始导入 stock_concepts_cache.xlsx 到数据库...")

    create_table_if_not_exists()

    success = import_stock_cache_from_excel(excel_path)

    if success:
        print("🎉 导入完成!")

        with get_session() as session:
            result = session.execute(text("SELECT COUNT(*) FROM stock_pools"))
            count = result.scalar()
            print(f"📊 数据库中现有 {count} 条股票缓存记录")

            result = session.execute(text("SELECT name, ts_code FROM stock_pools LIMIT 10"))
            print("📋 前 10 条股票缓存记录:")
            for row in result:
                print(f"   • {row.name}: {row.ts_code}")
    else:
        print("❌ 导入失败!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
