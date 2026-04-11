"""
Tushare 数据获取模块

职责：
1. 从 Tushare API 获取利润表、现金流量表、资产负债表
2. 写入 `financial_quarterly_data` 表（单季度格式）
3. 获取前十大流通股东数据，写入 `stock_top10_holders_tushare` 表
4. 支持股票池批量获取和增量更新
5. 提供命令行接口

用法（在项目根目录运行）：
    # 单股票获取财务报表
    python tushare_data/fetcher.py --ts_code 000657.SZ --start_date 20120101

    # 股票池批量获取财务报表（增量）
    python tushare_data/fetcher.py --pool --years 5

    # 单股票获取前十大流通股东
    python tushare_data/fetcher.py --top10 --ts_code 000657.SZ --start_date 20200101

    # 股票池批量获取前十大流通股东（增量）
    python tushare_data/fetcher.py --top10 --pool --years 5
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tushare_data.config import TS_TOKEN
from config import DATABASE_URL
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tushare.fetcher")

_db_engine = None


def get_db_engine():
    """获取PostgreSQL数据库引擎"""
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_recycle=3600)
    return _db_engine


def get_pro(token: str = TS_TOKEN):
    try:
        import tushare as ts
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("未安装 tushare。请先执行 `pip install tushare` 后再运行本脚本。") from e
    return ts.pro_api(token)


def _combine_report_types(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """合并两种报表类型，type2 优先，type1 补空值；各自先去重，再对 end_date 做最终去重"""
    df1 = dedup_latest(df1) if not df1.empty else df1
    df2 = dedup_latest(df2) if not df2.empty else df2
    if df1.empty and df2.empty:
        return pd.DataFrame()
    if df1.empty:
        out = df2.copy()
        out["report_type"] = 2
        return out
    if df2.empty:
        out = df1.copy()
        out["report_type"] = 1
        return out
    common_keys = ["ts_code", "end_date"]
    out = df2.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    extra_cols = [c for c in df1.columns if c not in out.columns]
    for col in extra_cols:
        out[col] = np.nan
    merged = out.merge(df1, on=common_keys, how="outer", suffixes=("", "_t1"))
    for col in numeric_cols:
        if col in merged.columns and f"{col}_t1" in merged.columns:
            mask = merged[col].isna()
            merged.loc[mask, col] = merged.loc[mask, f"{col}_t1"]
    t1_extra = [c for c in df1.columns if c not in df2.columns and c not in common_keys]
    for col in t1_extra:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged.get(f"{col}_t1", np.nan))
        elif f"{col}_t1" in merged.columns:
            merged[col] = merged[f"{col}_t1"]
    merged = merged[[c for c in out.columns if c in merged.columns]]
    merged["report_type"] = 2
    merged = dedup_latest(merged)
    return merged


def fetch_income(pro, ts_code: str, start_date: str) -> pd.DataFrame:
    fields = [
        "ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
        "total_revenue", "revenue", "oper_cost", "operate_profit",
        "ebit", "ebitda", "n_income", "n_income_attr_p", "rd_exp"
    ]
    flds = ",".join(fields)
    df1 = pro.income(ts_code=ts_code, start_date=start_date, report_type="1", fields=flds)
    df2 = pro.income(ts_code=ts_code, start_date=start_date, report_type="2", fields=flds)
    return _combine_report_types(df1, df2)


def fetch_cashflow(pro, ts_code: str, start_date: str) -> pd.DataFrame:
    fields = [
        "ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
        "n_cashflow_act", "free_cashflow", "c_fr_sale_sg", "c_pay_acq_const_fiolta"
    ]
    flds = ",".join(fields)
    df1 = pro.cashflow(ts_code=ts_code, start_date=start_date, report_type="1", fields=flds)
    df2 = pro.cashflow(ts_code=ts_code, start_date=start_date, report_type="2", fields=flds)
    return _combine_report_types(df1, df2)


def fetch_balancesheet(pro, ts_code: str, start_date: str) -> pd.DataFrame:
    fields = [
        "ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
        "total_assets", "accounts_receiv", "inventories", "accounts_pay",
        "contract_liab", "total_hldr_eqy_exc_min_int"
    ]
    flds = ",".join(fields)
    df1 = pro.balancesheet(ts_code=ts_code, start_date=start_date, report_type="1", fields=flds)
    df2 = pro.balancesheet(ts_code=ts_code, start_date=start_date, report_type="2", fields=flds)
    return _combine_report_types(df1, df2)


def fetch_top10_holders(pro, ts_code: str, start_date: str) -> pd.DataFrame:
    fields = [
        "ts_code", "ann_date", "end_date",
        "holder_name", "holder_type",
        "hold_amount", "hold_ratio", "hold_float_ratio", "hold_change"
    ]
    return pro.top10_holders(
        ts_code=ts_code, start_date=start_date,
        fields=",".join(fields)
    )


def assign_holder_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_end_date_dt"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values(["_end_date_dt", "ann_date", "holder_name"], na_position="first")
    df["_rank"] = df.groupby(["ts_code", "end_date"])["hold_amount"].rank(
        method="first", ascending=False
    )
    df["holder_rank"] = df["_rank"].astype(int)
    df = df.drop(columns=["_end_date_dt", "_rank"])
    return df


def get_existing_top10_dates(ts_code: str, engine) -> Set[str]:
    sql = "SELECT report_date FROM stock_top10_holders_tushare WHERE ts_code = :ts_code"
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"ts_code": ts_code})
        return {row[0] for row in result.fetchall()}


def get_all_existing_top10_counts(engine) -> dict:
    """返回 {ts_code: set(existing_report_dates)}，全场只查一次"""
    sql = "SELECT ts_code, report_date FROM stock_top10_holders_tushare"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    if df.empty:
        return {}
    return df.groupby("ts_code")["report_date"].apply(set).to_dict()


def upsert_top10_to_db(df: pd.DataFrame, engine,
                        existing_dates: Set[str] = None,
                        stock_name: str = None) -> int:
    if existing_dates is None:
        existing_dates = set()

    df = assign_holder_rank(df)

    if df.empty:
        return 0

    df["report_date"] = df["end_date"]
    if stock_name:
        df["stock_name"] = stock_name

    df["report_date"] = pd.to_datetime(df["report_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["report_date"].notna()]
    if df.empty:
        return 0

    df["report_date_str"] = df["report_date"].dt.strftime("%Y%m%d")
    df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["ann_date"] = df["ann_date"].dt.strftime("%Y%m%d")

    if existing_dates:
        before = len(df)
        df = df[~df["report_date_str"].isin(existing_dates)]
        logger.debug(f"跳过 {before - len(df)} 条已有数据")

    db_columns = [
        "ts_code", "stock_name", "report_date", "ann_date", "holder_rank",
        "holder_name", "holder_type",
        "hold_amount", "hold_ratio", "hold_float_ratio", "hold_change"
    ]

    df_out = df[[c for c in db_columns if c in df.columns]].copy()
    if df_out.empty:
        return 0

    non_key_cols = [c for c in db_columns if c not in ["ts_code", "report_date", "holder_rank"]]
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key_cols])
    col_clause = ", ".join(db_columns)
    placeholders = ", ".join([f":{c}" for c in db_columns])

    sql = f"""
        INSERT INTO stock_top10_holders_tushare ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT (ts_code, report_date, holder_rank) DO UPDATE SET
            {update_clause},
            updated_at = CURRENT_TIMESTAMP
    """

    records = []
    for _, row in df_out.iterrows():
        record = {}
        for col in db_columns:
            val = row.get(col)
            if col == "report_date":
                if isinstance(val, pd.Timestamp):
                    record[col] = val.strftime("%Y%m%d")
                elif isinstance(val, str):
                    record[col] = val
                elif pd.notna(val):
                    record[col] = str(val)[:8]
                else:
                    record[col] = None
            elif col == "ann_date":
                if isinstance(val, pd.Timestamp):
                    record[col] = val.strftime("%Y%m%d")
                elif pd.notna(val):
                    record[col] = str(val)[:8]
                else:
                    record[col] = None
            elif col in row.index and not pd.isna(val):
                if isinstance(val, (np.integer, np.floating)):
                    record[col] = float(val)
                else:
                    record[col] = val
            else:
                record[col] = None
        records.append(record)

    with engine.connect() as conn:
        conn.execute(text(sql), records)
        conn.commit()

    return len(records)


def fetch_and_save_top10(ts_code: str, start_date: str, token: str = TS_TOKEN,
                         existing_dates: Set[str] = None,
                         stock_name: str = None) -> dict:
    pro = get_pro(token)
    df = fetch_top10_holders(pro, ts_code, start_date)

    engine = get_db_engine()
    count = upsert_top10_to_db(df, engine, existing_dates, stock_name)

    result = {
        "ts_code": ts_code,
        "fetched_rows": len(df),
        "saved_rows": count,
    }
    return result


def dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_end_date_dt"] = pd.to_datetime(tmp["end_date"].astype(str), format="%Y%m%d", errors="coerce")
    sort_cols = ["_end_date_dt", "ann_date", "f_ann_date"]
    sort_cols = [c for c in sort_cols if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols, na_position="first")
    if "ts_code" in tmp.columns:
        tmp = tmp.groupby(["ts_code", "end_date"], as_index=False).tail(1)
    else:
        tmp = tmp.groupby("end_date", as_index=False).tail(1)
    result = tmp.sort_values("_end_date_dt", na_position="first").reset_index(drop=True)
    result = result.drop(columns=["_end_date_dt"])
    return result


def get_stock_pool_from_db(engine) -> List[Tuple[str, str]]:
    sql = "SELECT ts_code, name FROM stock_pools ORDER BY ts_code"
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [(row[0], row[1]) for row in result.fetchall()]


def get_existing_report_dates(ts_code: str, engine) -> Set[str]:
    sql = "SELECT end_date FROM financial_quarterly_data WHERE ts_code = :ts_code"
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"ts_code": ts_code})
        return {row[0] for row in result.fetchall()}


def upsert_quarterly_to_db(income_df: pd.DataFrame, cash_df: pd.DataFrame,
                            bs_df: pd.DataFrame, engine,
                            existing_dates: Set[str] = None) -> int:
    if existing_dates is None:
        existing_dates = set()

    income = dedup_latest(income_df)
    cash = dedup_latest(cash_df)
    bs = dedup_latest(bs_df)

    df = income.merge(
        cash.drop(columns=["report_type", "ann_date", "f_ann_date"], errors="ignore"),
        on=["ts_code", "end_date"], how="outer"
    ).merge(
        bs.drop(columns=["report_type", "ann_date", "f_ann_date"], errors="ignore"),
        on=["ts_code", "end_date"], how="outer"
    )

    df["end_date"] = pd.to_datetime(df["end_date"].astype(str), format="%Y%m%d", errors="coerce")
    if "ann_date" in df.columns:
        df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    if "f_ann_date" in df.columns:
        df["f_ann_date"] = pd.to_datetime(df["f_ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)

    date_cols = {"end_date", "ann_date", "f_ann_date", "report_date"}
    for col in df.columns:
        if col not in ["ts_code", "name"] and col not in date_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["end_date"].notna()]
    if df.empty:
        return 0

    db_columns = [
        "ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
        "total_revenue", "revenue", "oper_cost", "operate_profit",
        "ebit", "ebitda", "n_income", "n_income_attr_p", "rd_exp",
        "n_cashflow_act", "free_cashflow", "c_fr_sale_sg", "c_pay_acq_const_fiolta",
        "total_assets", "accounts_receiv", "inventories", "accounts_pay",
        "contract_liab", "total_hldr_eqy_exc_min_int",
    ]

    df["end_date_str"] = df["end_date"].dt.strftime("%Y%m%d")

    if existing_dates:
        before_count = len(df)
        df = df[~df["end_date_str"].isin(existing_dates)]
        after_count = len(df)
        logger.debug(f"跳过 {before_count - after_count} 条已有数据")

    df = df[[c for c in db_columns if c in df.columns]]
    if df.empty:
        return 0

    non_key_cols = [c for c in db_columns if c not in ["ts_code", "end_date", "report_type"]]
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key_cols])
    col_clause = ", ".join(db_columns)
    placeholders = ", ".join([f":{c}" for c in db_columns])

    sql = f"""
        INSERT INTO financial_quarterly_data ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT (ts_code, end_date, report_type) DO UPDATE SET
            {update_clause},
            updated_at = CURRENT_TIMESTAMP
    """

    records = []
    for idx, row in df.iterrows():
        record = {}
        end_date_val = row.get("end_date")
        end_date_str = end_date_val.strftime("%Y%m%d") if pd.notna(end_date_val) else None
        ann_date_val = row.get("ann_date")
        ann_date_str = ann_date_val.strftime("%Y%m%d") if pd.notna(ann_date_val) else None
        f_ann_date_val = row.get("f_ann_date")
        f_ann_date_str = f_ann_date_val.strftime("%Y%m%d") if pd.notna(f_ann_date_val) else None
        for col in db_columns:
            if col == "end_date":
                record[col] = end_date_str
            elif col == "ann_date":
                record[col] = ann_date_str
            elif col == "f_ann_date":
                record[col] = f_ann_date_str
            elif col in row.index and not pd.isna(row.get(col)):
                val = row.get(col)
                if isinstance(val, (np.integer, np.floating)):
                    record[col] = float(val)
                else:
                    record[col] = val
            else:
                record[col] = None
        records.append(record)

    with engine.connect() as conn:
        conn.execute(text(sql), records)
        conn.commit()

    return len(records)


def fetch_and_save(ts_code: str, start_date: str, token: str = TS_TOKEN,
                   existing_dates: Set[str] = None) -> dict:
    pro = get_pro(token)

    income = fetch_income(pro, ts_code, start_date)
    cash = fetch_cashflow(pro, ts_code, start_date)
    bs = fetch_balancesheet(pro, ts_code, start_date)

    engine = get_db_engine()
    count = upsert_quarterly_to_db(income, cash, bs, engine, existing_dates)

    result = {
        "ts_code": ts_code,
        "income_rows": len(income),
        "cash_rows": len(cash),
        "bs_rows": len(bs),
        "saved_rows": count,
    }
    return result


def get_all_existing_report_counts(engine) -> dict:
    """返回 {ts_code: set(existing_report_dates)}，全场只查一次"""
    sql = "SELECT ts_code, end_date FROM financial_quarterly_data"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    if df.empty:
        return {}
    return df.groupby("ts_code")["end_date"].apply(set).to_dict()


def fetch_stock_pool(years: int = 5, incremental: bool = True,
                     batch_size: int = 100) -> dict:
    from dateutil.relativedelta import relativedelta

    engine = get_db_engine()

    stock_pool = get_stock_pool_from_db(engine)
    if not stock_pool:
        logger.warning("股票池为空，请先更新 stock_pools 表")
        return {"success": 0, "skipped": 0, "failed": 0, "failed_stocks": []}

    start_date = (datetime.now() - relativedelta(years=years)).strftime("%Y0101")
    logger.info(f"开始获取 {len(stock_pool)} 只股票最近 {years} 年财务数据")
    logger.info(f"起始日期: {start_date}, 增量模式: {incremental}")

    all_existing = get_all_existing_report_counts(engine) if incremental else {}
    logger.info(f"已有数据股票数: {len(all_existing)}")

    if incremental:
        stock_pool = [
            (ts_code, name) for ts_code, name in stock_pool
            if len(all_existing.get(ts_code, set())) < 20
        ]
        logger.info(f"需要获取的股票数: {len(stock_pool)}")

    results = {"success": 0, "skipped": 0, "failed": 0, "failed_stocks": []}

    pbar = tqdm(stock_pool, desc="获取财务数据", ncols=100)
    for ts_code, name in pbar:
        pbar.set_postfix_str(ts_code)

        existing_dates = all_existing.get(ts_code, set()) if incremental else set()

        if incremental and len(existing_dates) >= 20:
            results["skipped"] += 1
            logger.debug(f"{ts_code} 已有 {len(existing_dates)} 个报告期，跳过")
            continue

        saved = False
        for attempt in range(5):
            try:
                result = fetch_and_save(ts_code, start_date, existing_dates=existing_dates)
                if result["saved_rows"] > 0:
                    results["success"] += 1
                elif incremental:
                    results["skipped"] += 1
                saved = True
                break
            except Exception as e:
                if attempt < 4:
                    logger.warning(f"获取 {ts_code} 失败，10秒后重试 ({attempt + 1}/5): {e}")
                    time.sleep(10)
                else:
                    results["failed"] += 1
                    results["failed_stocks"].append(ts_code)
                    logger.error(f"获取 {ts_code} 失败，已重试5次: {e}")

        time.sleep(0.2)

    logger.info(f"完成: 成功 {results['success']}, 跳过 {results['skipped']}, 失败 {results['failed']}")
    if results["failed_stocks"]:
        logger.info(f"失败股票: {results['failed_stocks'][:10]}")

    return results


def fetch_top10_pool(years: int = 5, incremental: bool = True,
                      batch_size: int = 100, limit: int = None) -> dict:
    from datasource.models import get_create_sql
    from dateutil.relativedelta import relativedelta

    engine = get_db_engine()

    with engine.connect() as conn:
        try:
            for stmt in get_create_sql("stock_top10_holders_tushare").split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()
        except Exception as e:
            logger.debug(f"建表跳过（或已存在）: {e}")
            conn.execute(text("ROLLBACK"))
            conn.commit()

    stock_pool = get_stock_pool_from_db(engine)
    if not stock_pool:
        logger.warning("股票池为空，请先更新 stock_pools 表")
        return {"success": 0, "skipped": 0, "failed": 0, "failed_stocks": []}
    if limit:
        stock_pool = stock_pool[:limit]

    start_date = (datetime.now() - relativedelta(years=years)).strftime("%Y0101")
    logger.info(f"开始获取 {len(stock_pool)} 只股票最近 {years} 年前十大流通股东数据")
    logger.info(f"起始日期: {start_date}, 增量模式: {incremental}")

    all_existing = get_all_existing_top10_counts(engine) if incremental else {}
    logger.info(f"已有数据股票数: {len(all_existing)}")

    if incremental:
        stock_pool = [
            (ts_code, name) for ts_code, name in stock_pool
            if len(all_existing.get(ts_code, set())) < 20
        ]
        logger.info(f"需要获取的股票数: {len(stock_pool)}")

    results = {"success": 0, "skipped": 0, "failed": 0, "failed_stocks": []}

    pbar = tqdm(stock_pool, desc="获取Top10股东", ncols=100)
    for ts_code, name in pbar:
        pbar.set_postfix_str(ts_code)

        existing_dates = all_existing.get(ts_code, set())

        if incremental and len(existing_dates) >= 20:
            results["skipped"] += 1
            logger.debug(f"{ts_code} 已有 {len(existing_dates)} 个报告期，跳过")
            continue

        saved = False
        for attempt in range(5):
            try:
                result = fetch_and_save_top10(
                    ts_code, start_date, existing_dates=existing_dates, stock_name=name
                )
                if result["saved_rows"] > 0:
                    results["success"] += 1
                elif incremental:
                    results["skipped"] += 1
                saved = True
                break
            except Exception as e:
                if attempt < 4:
                    logger.warning(f"获取 {ts_code} 失败，10秒后重试 ({attempt + 1}/5): {e}")
                    time.sleep(10)
                else:
                    results["failed"] += 1
                    results["failed_stocks"].append(ts_code)
                    logger.error(f"获取 {ts_code} 失败，已重试5次: {e}")

        time.sleep(0.2)

    logger.info(f"完成: 成功 {results['success']}, 跳过 {results['skipped']}, 失败 {results['failed']}")
    if results["failed_stocks"]:
        logger.info(f"失败股票: {results['failed_stocks'][:10]}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Tushare 数据获取")
    parser.add_argument("--ts_code", type=str, help="股票代码（如 000657.SZ）")
    parser.add_argument("--start_date", type=str, default="20120101", help="起始日期（如 20120101）")
    parser.add_argument("--token", type=str, default=TS_TOKEN, help="Tushare token")
    parser.add_argument("--pool", action="store_true", help="批量获取股票池数据")
    parser.add_argument("--years", type=int, default=5, help="获取最近 N 年数据（默认 5）")
    parser.add_argument("--no-incremental", action="store_true", help="禁用增量模式（全量获取）")
    parser.add_argument("--top10", action="store_true", help="获取前十大流通股东数据")
    parser.add_argument("--limit", type=int, default=None, help="限制股票池数量（用于测试）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.top10:
        if args.pool:
            incremental = not args.no_incremental
            results = fetch_top10_pool(years=args.years, incremental=incremental, limit=args.limit)
            print(f"成功: {results['success']}, 跳过: {results['skipped']}, 失败: {results['failed']}")
            if results["failed_stocks"]:
                print(f"失败股票（前10个）: {results['failed_stocks'][:10]}")
        elif args.ts_code:
            result = fetch_and_save_top10(args.ts_code, args.start_date, args.token)
            print(f"已保存 {result['saved_rows']} 条记录: {result['ts_code']}")
        else:
            print("前十大流通股东模式请指定 --ts_code 或 --pool")
            return 1
    elif args.pool:
        incremental = not args.no_incremental
        results = fetch_stock_pool(years=args.years, incremental=incremental)
        print(f"成功: {results['success']}, 跳过: {results['skipped']}, 失败: {results['failed']}")
        if results["failed_stocks"]:
            print(f"失败股票（前10个）: {results['failed_stocks'][:10]}")
    elif args.ts_code:
        result = fetch_and_save(args.ts_code, args.start_date, args.token)
        print(f"已保存 {result['saved_rows']} 条记录: {result['ts_code']}")
    else:
        print("请指定 --ts_code 或 --pool")
        return 1

    return 0


INDEX_CODES = {"000001.SH", "000300.SH", "000905.SH", "000016.SH", "000688.SH",
                "399001.SZ", "399006.SZ", "399005.SZ", "399300.SZ", "399673.SZ"}


def fetch_market_data(
    ts_code: str,
    start_date: str,
    end_date: str,
    fqt: int = 0,
    max_retry: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """
    使用 Tushare Pro 获取日线行情（个股或指数）。

    参数:
        ts_code: 股票/指数代码，如 000001.SZ / 600519.SH / 000300.SH
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        fqt: 复权类型（暂不支持，填0）
        max_retry: 最大重试次数
        retry_delay: 重试间隔（秒）

    返回:
        DataFrame，index=date，columns=[open, high, low, close, vol, turnover, turnover_rate]
        空DataFrame表示获取失败
    """
    import time
    pro = get_pro()
    is_index = ts_code in INDEX_CODES
    for attempt in range(max_retry):
        try:
            if is_index:
                df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception:
            if attempt < max_retry - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return pd.DataFrame()
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            if attempt < max_retry - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return pd.DataFrame()
        break

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={
        "trade_date": "date",
        "amount": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date")
    for c in ["open", "high", "low", "close", "vol", "turnover", "turnover_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep_cols = ["open", "high", "low", "close", "vol", "turnover", "turnover_rate"]
    out_cols = [c for c in keep_cols if c in df.columns]
    df = df[out_cols].copy()
    return df


def fetch_market_data_batch(
    ts_code_list: List[str],
    start_date: str,
    end_date: str,
    fqt: int = 0,
    max_retry: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    """
    批量获取多只股票的日线行情。

    参数:
        ts_code_list: 股票代码列表，如 ["600519.SH", "600036.SH"]
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        fqt: 复权类型（暂不支持，填0）
        max_retry: 最大重试次数
        retry_delay: 重试间隔（秒）

    返回:
        Dict[str, DataFrame]，key=ts_code，value=DataFrame(index=date, columns=[open,high,low,close,vol,turnover,turnover_rate])
        失败的单只返回空 DataFrame（不整体报错）
    """
    import time
    result: Dict[str, pd.DataFrame] = {}

    for ts_code in ts_code_list:
        df = fetch_market_data(ts_code, start_date, end_date, fqt, max_retry, retry_delay)
        result[ts_code] = df

    return result


if __name__ == "__main__":
    sys.exit(main())