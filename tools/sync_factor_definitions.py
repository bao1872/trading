#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子定义同步脚本

Purpose: 将 factor_lib 注册表中的因子定义同步到数据库 factor_definition 表
Inputs: factor_lib.FACTOR_REGISTRY
Outputs: 数据库 factor_definition 表更新
How to Run:
    python tools/sync_factor_definitions.py
    python tools/sync_factor_definitions.py --dry-run
Examples:
    python tools/sync_factor_definitions.py
Side Effects: 更新/插入 factor_definition 表记录
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from datasource.database import get_engine, table_exists


def sync_factor_definitions(dry_run: bool = False):
    """同步因子定义到数据库"""
    from factor_lib import FACTOR_REGISTRY

    print("=" * 60)
    print(f"同步因子定义到数据库 (共 {len(FACTOR_REGISTRY)} 个因子)")
    print("=" * 60)

    records = []
    for name, meta in FACTOR_REGISTRY.items():
        records.append({
            "factor_name": name,
            "factor_group": meta.get("category", ""),
            "freq_type_supported": "1d,1w",  # 默认支持日线和周线
            "description": meta.get("description", ""),
            "source_module": meta.get("source_module", ""),
            "source_function": meta.get("source_function", ""),
            "direction": meta.get("direction", "neutral"),
            "is_core": meta.get("is_core", False),
            "is_active": True,
        })

    df = pd.DataFrame(records)

    if dry_run:
        print("[DRY-RUN] 将同步以下因子:")
        print(df[["factor_name", "factor_group", "description"]].to_string(index=False))
        return

    engine = get_engine()
    with engine.begin() as conn:
        if not table_exists(conn, "factor_definition"):
            print("[FAIL] factor_definition 表不存在，请先运行 init_db_schema.py")
            return

        # 使用 upsert 方式写入
        from datasource.database import bulk_upsert
        bulk_upsert(conn, "factor_definition", df, unique_keys=["factor_name"])

    print(f"[OK] 同步完成: {len(df)} 个因子")


def main():
    parser = argparse.ArgumentParser(description="同步因子定义到数据库")
    parser.add_argument("--dry-run", action="store_true", help="试运行")
    args = parser.parse_args()
    sync_factor_definitions(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
