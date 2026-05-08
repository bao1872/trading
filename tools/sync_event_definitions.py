#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件定义同步脚本

Purpose: 将 event_lib 注册表中的事件定义同步到数据库 event_definition 和 event_factor_map 表
Inputs: event_lib.EVENT_REGISTRY
Outputs: 数据库 event_definition / event_factor_map 表更新
How to Run:
    python tools/sync_event_definitions.py
    python tools/sync_event_definitions.py --dry-run
Examples:
    python tools/sync_event_definitions.py
Side Effects: 更新/插入 event_definition 和 event_factor_map 表记录
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasource.database import get_engine, table_exists, bulk_upsert


def sync_event_definitions(dry_run: bool = False):
    """同步事件定义到数据库"""
    from event_lib import EVENT_REGISTRY

    print("=" * 60)
    print(f"同步事件定义到数据库 (共 {len(EVENT_REGISTRY)} 个事件)")
    print("=" * 60)

    event_records = []
    map_records = []

    for name, meta in EVENT_REGISTRY.items():
        event_records.append({
            "event_name": name,
            "event_group": meta.get("category", ""),
            "description": meta.get("description", ""),
            "required_factors": ",".join(meta.get("required_factors", [])),
            "freq_type_supported": "1d,1w",
            "is_active": True,
        })

        for factor_name in meta.get("required_factors", []):
            map_records.append({
                "event_name": name,
                "factor_name": factor_name,
                "role": "trigger",  # 默认 role 为 trigger
            })

    event_df = pd.DataFrame(event_records)
    map_df = pd.DataFrame(map_records)

    if dry_run:
        print("[DRY-RUN] 将同步以下事件:")
        print(event_df[["event_name", "event_group", "description"]].to_string(index=False))
        print(f"\n[DRY-RUN] 事件-因子映射: {len(map_df)} 条")
        return

    engine = get_engine()

    # 使用普通连接（非上下文管理器），因为 bulk_upsert 内部会自行 commit
    conn = engine.connect()
    try:
        if not table_exists(conn, "event_definition"):
            print("[FAIL] event_definition 表不存在，请先运行 init_db_schema.py")
            return

        bulk_upsert(conn, "event_definition", event_df, unique_keys=["event_name"])
        print(f"[OK] event_definition 同步完成: {len(event_df)} 个事件")

        if map_records:
            bulk_upsert(conn, "event_factor_map", map_df, unique_keys=["event_name", "factor_name"])
            print(f"[OK] event_factor_map 同步完成: {len(map_df)} 条映射")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="同步事件定义到数据库")
    parser.add_argument("--dry-run", action="store_true", help="试运行")
    args = parser.parse_args()
    sync_event_definitions(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
