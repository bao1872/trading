#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入 stock.xlsx 到数据库

Purpose: 将 stock.xlsx 文件中的股票列表导入到数据库中
Inputs: stock.xlsx 文件（项目根目录下）
Outputs: stock_list 表中的数据
How to Run: 
    # 从项目根目录运行
    python app/import_stock_xlsx_to_db.py
    
    # 或使用模块方式运行
    python -m app.import_stock_xlsx_to_db
Side Effects: 向 stock_list 表插入/更新数据
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datasource.database import get_session
from app.models import get_create_sql
from sqlalchemy import text


def create_table_if_not_exists():
    """创建表（如果不存在）"""
    with get_session() as session:
        sql = get_create_sql("stock_list")
        session.execute(text(sql))
        session.commit()
        print("✅ 已确保 stock_list 表存在")


def import_stock_list_from_excel(excel_path: str):
    """从 Excel 导入股票列表到数据库"""
    # 读取 Excel 文件
    df = pd.read_excel(excel_path)
    print(f"📊 从 {excel_path} 读取到 {len(df)} 条股票记录")
    print(f"📋 列名: {df.columns.tolist()}")
    
    # 检查是否存在 '股票名称' 列
    if '股票名称' not in df.columns:
        print(f"❌ 错误: Excel 文件中不存在 '股票名称' 列")
        print(f"✅ 可用列: {df.columns.tolist()}")
        return False
    
    # 只保留股票名称列，ts_code 会在后续过程中填充（如果需要的话）
    df = df[['股票名称']].copy()
    df.rename(columns={'股票名称': 'stock_name'}, inplace=True)
    df['ts_code'] = None  # 添加空列
    
    # 去重
    original_count = len(df)
    df.drop_duplicates(subset=['stock_name'], keep='first', inplace=True)
    if len(df) < original_count:
        print(f"⚠️  去重: 从 {original_count} 条记录去重至 {len(df)} 条")
    
    print(f"📈 最终导入 {len(df)} 条不重复的股票记录")
    
    # 使用 upsert 插入数据库
    with get_session() as session:
        # 删除现有数据（可选，根据需要决定是否清空）
        session.execute(text("DELETE FROM stock_list"))
        print("🗑️  已清空现有股票列表")
        
        # 批量插入
        records = df.to_dict(orient='records')
        for record in records:
            sql = """
                INSERT INTO stock_list (stock_name, ts_code, created_at, updated_at)
                VALUES (:stock_name, :ts_code, NOW(), NOW())
                ON CONFLICT (stock_name) DO UPDATE SET
                    ts_code = COALESCE(EXCLUDED.ts_code, stock_list.ts_code),
                    updated_at = NOW()
            """
            session.execute(text(sql), record)
        
        session.commit()
        print(f"✅ 成功导入 {len(records)} 条股票记录到数据库")
    
    return True


def main():
    """主函数"""
    excel_path = "stock.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"❌ 错误: 文件 {excel_path} 不存在")
        return 1
    
    print("🚀 开始导入 stock.xlsx 到数据库...")
    
    # 创建表（如果不存在）
    create_table_if_not_exists()
    
    # 导入数据
    success = import_stock_list_from_excel(excel_path)
    
    if success:
        print("🎉 导入完成!")
        
        # 显示导入结果
        with get_session() as session:
            result = session.execute(text("SELECT COUNT(*) FROM stock_list"))
            count = result.scalar()
            print(f"📊 数据库中现有 {count} 条股票记录")
            
            # 显示前几条记录
            result = session.execute(text("SELECT stock_name FROM stock_list LIMIT 10"))
            print("📋 前 10 条股票记录:")
            for row in result:
                print(f"   • {row.stock_name}")
    else:
        print("❌ 导入失败!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())