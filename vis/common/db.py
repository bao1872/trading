# -*- coding: utf-8 -*-
"""
Purpose:
    数据库连接封装，供 vis/ 下 Streamlit 应用共享。
    复用 datasource.database 的 SSOT 接口。

Inputs:
    - config.DATABASE_URL

Outputs:
    - get_session(): 上下文管理器，获取数据库会话
    - query_sql(): 执行原始 SQL 返回 DataFrame
    - get_stock_name_map(): 获取 ts_code -> name 映射

How to Run:
    无需单独运行，被其他模块 import

Examples:
    from vis.common.db import get_stock_name_map
    names = get_stock_name_map()

Side Effects:
    - 读取数据库（stock_pools 表）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datasource.database import get_session, query_sql


def get_stock_name_map() -> dict:
    """获取 ts_code -> 股票名称 映射字典"""
    with get_session() as session:
        df = query_sql(session, "SELECT ts_code, name FROM stock_pools")
    if df.empty:
        return {}
    return dict(zip(df["ts_code"], df["name"]))


if __name__ == "__main__":
    names = get_stock_name_map()
    print(f"股票名称映射: {len(names)} 只")
    sample_keys = list(names.keys())[:5]
    for k in sample_keys:
        print(f"  {k}: {names[k]}")
