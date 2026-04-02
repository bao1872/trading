# -*- coding: utf-8 -*-
"""
数据导入脚本 - 将 pickle 数据集导入 PostgreSQL 数据库

用途：
    将 dataset/*.pkl 中的 K 线数据导入 PostgreSQL 数据库

支持数据集：
    - daily_500.pkl -> stock_k_data (freq='d')
    - weekly_500.pkl -> stock_k_data (freq='w')
    - min60_2000.pkl -> stock_k_data (freq='60m')
    - min15_8000.pkl -> stock_k_data (freq='15m')

运行命令：
    python datasource/import_pkl_to_db.py
    python datasource/import_pkl_to_db.py --freq 5m
    python datasource/import_pkl_to_db.py --limit 10

注意：
    这是一个一次性导入脚本，导入完成后可以直接从数据库读取
"""
import os
import pickle
import sys
from typing import Dict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_DIR, DATA_DIR, DATABASE_URL
from datasource.database import get_session, table_exists, bulk_upsert
from datasource.init_db import create_database_dir, init_tables


DATASET_FILES = {
    "d": ("daily_500.pkl", "d"),
    "w": ("weekly_500.pkl", "w"),
    "60m": ("min60_2000.pkl", "60m"),
    "15m": ("min15_8000.pkl", "15m"),
    "5m": ("min15_8000.pkl", "5m"),
}


def _symbol_to_code(ts_code: str) -> str:
    """ts_code 如 600489.SH -> 600489"""
    return ts_code.split(".")[0]


def _code_to_symbol(code: str) -> str:
    """code 如 600489 -> 600489.SH"""
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def load_pkl_file(filepath: str) -> Dict[str, Dict]:
    """加载 pickle 文件"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return {}
    with open(filepath, "rb") as f:
        return pickle.load(f)


def import_dataset(freq: str, limit_stocks: int = None, batch_size: int = 5000):
    """导入指定频率的数据集"""
    if freq not in DATASET_FILES:
        print(f"不支持的频率: {freq}")
        print(f"支持的频率: {list(DATASET_FILES.keys())}")
        return

    pkl_file, db_freq = DATASET_FILES[freq]
    filepath = os.path.join(DATASET_DIR, pkl_file)

    print(f"\n{'=' * 60}")
    print(f"导入 {freq} 数据 ({pkl_file})")
    print(f"{'=' * 60}")

    data_cache = load_pkl_file(filepath)
    if not data_cache:
        print("无数据可导入")
        return

    stock_codes = list(data_cache.keys())
    if limit_stocks:
        stock_codes = stock_codes[:limit_stocks]

    print(f"股票数量: {len(stock_codes)}")

    total_records = 0
    with get_session() as session:
        for code in tqdm(stock_codes, desc=f"导入{freq}"):
            stock_data = data_cache[code]
            name = stock_data.get("name", "")
            df = stock_data.get("data")

            if df is None or df.empty:
                continue

            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.isna()]
            df = df.sort_index()

            df["ts_code"] = _code_to_symbol(code)
            df["freq"] = db_freq
            df["bar_time"] = df.index
            df["name"] = name

            df_for_db = df.reset_index(drop=True)
            df_for_db = df_for_db[["ts_code", "freq", "bar_time", "open", "high", "low", "close", "volume"]]

            df_for_db["bar_time"] = pd.to_datetime(df_for_db["bar_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            try:
                bulk_upsert(session, "stock_k_data", df_for_db, unique_keys=["ts_code", "freq", "bar_time"], batch_size=batch_size)
                total_records += len(df_for_db)
            except Exception as e:
                print(f"\n导入 {code} 失败: {e}")

    print(f"\n✅ 导入完成: {freq}, 共 {total_records} 条记录")
    return total_records


def import_all(limit_stocks: int = None):
    """导入所有数据集"""
    print(f"\n{'=' * 60}")
    print("导入所有数据集到 PostgreSQL")
    print(f"{'=' * 60}")
    print(f"数据集目录: {DATASET_DIR}")
    print(f"数据库: {DATABASE_URL}")
    print(f"{'=' * 60}")

    create_database_dir()
    init_tables()

    total_records = 0
    for freq in ["d", "w", "60m", "15m"]:
        records = import_dataset(freq, limit_stocks=limit_stocks)
        if records:
            total_records += records

    print(f"\n{'=' * 60}")
    print(f"✅ 全部导入完成, 共 {total_records} 条记录")
    print(f"{'=' * 60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="将 pickle 数据集导入 PostgreSQL 数据库")
    parser.add_argument("--freq", type=str, default=None,
                        help="指定频率: d, w, 60m, 15m, 5m (不指定则导入所有)")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制导入的股票数量（用于测试）")
    args = parser.parse_args()

    if args.freq:
        import_dataset(args.freq, limit_stocks=args.limit)
    else:
        import_all(limit_stocks=args.limit)


if __name__ == "__main__":
    main()