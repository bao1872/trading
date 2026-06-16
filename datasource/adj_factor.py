# -*- coding: utf-8 -*-
"""
复权因子管理模块 - 从 tushare 获取复权因子并存储到数据库

Purpose:
    管理股票复权因子（adj_factor），用于将不复权价格转换为前复权价格

Inputs:
    tushare adj_factor 接口

Outputs:
    stock_adj_factor 表（ts_code, trade_date, adj_factor）

How to Run:
    python datasource/adj_factor.py                        # 全量回补所有股票
    python datasource/adj_factor.py --ts_code 688678.SH    # 单只股票
    python datasource/adj_factor.py --incremental           # 增量更新（只拉取缺失部分）
    python datasource/adj_factor.py --test 688678.SH       # 测试单只股票前复权验证

Side Effects:
    写入 stock_adj_factor 表（幂等：upsert 模式）
"""
import sys
import os
import argparse
import time

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, bulk_upsert

ADJ_FACTOR_TABLE = "stock_adj_factor"

ADJ_FACTOR_DDL = """
CREATE TABLE IF NOT EXISTS stock_adj_factor (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    adj_factor FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_adj_factor_ts_code ON stock_adj_factor(ts_code);
CREATE INDEX IF NOT EXISTS idx_adj_factor_trade_date ON stock_adj_factor(trade_date);
CREATE INDEX IF NOT EXISTS idx_adj_factor_ts_date ON stock_adj_factor(ts_code, trade_date);
"""


def ensure_adj_factor_table():
    """确保 stock_adj_factor 表存在"""
    from datasource.database import get_engine
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(ADJ_FACTOR_DDL))


def fetch_adj_factor(ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    从 tushare 获取单只股票的复权因子

    Args:
        ts_code: 股票代码，如 '688678.SH'
        start_date: 起始日期，如 '20200101'
        end_date: 结束日期，如 '20260530'

    Returns:
        DataFrame with columns: ts_code, trade_date, adj_factor
    """
    from tushare_data.config import TS_TOKEN
    import tushare as ts

    pro = ts.pro_api(TS_TOKEN)
    kwargs = {"ts_code": ts_code}
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date

    df = pro.adj_factor(**kwargs)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[["ts_code", "trade_date", "adj_factor"]]
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def save_adj_factor(df: pd.DataFrame) -> int:
    """
    保存复权因子到数据库（upsert 模式）

    Args:
        df: DataFrame with columns: ts_code, trade_date, adj_factor

    Returns:
        保存的记录数
    """
    if df.empty:
        return 0

    ensure_adj_factor_table()

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

    with get_session() as session:
        bulk_upsert(session, ADJ_FACTOR_TABLE, df, unique_keys=["ts_code", "trade_date"])

    return len(df)


def fetch_and_save_adj_factor(ts_code: str, start_date: str = None, end_date: str = None) -> int:
    """
    获取并保存单只股票的复权因子

    Args:
        ts_code: 股票代码
        start_date: 起始日期
        end_date: 结束日期

    Returns:
        保存的记录数
    """
    df = fetch_adj_factor(ts_code, start_date, end_date)
    if df.empty:
        return 0
    return save_adj_factor(df)


def batch_fetch_adj_factor(ts_codes: list, start_date: str = None, end_date: str = None,
                           sleep_sec: float = 0.3) -> int:
    """
    批量获取并保存复权因子

    Args:
        ts_codes: 股票代码列表
        start_date: 起始日期
        end_date: 结束日期
        sleep_sec: 每次请求间隔（秒），避免触发tushare频率限制

    Returns:
        总保存记录数
    """
    from tqdm import tqdm

    total = 0
    for ts_code in tqdm(ts_codes, desc="获取复权因子", unit="只"):
        try:
            cnt = fetch_and_save_adj_factor(ts_code, start_date, end_date)
            total += cnt
        except Exception as e:
            print(f"  {ts_code} 获取失败: {e}")
        time.sleep(sleep_sec)

    return total


def incremental_update_adj_factor(ts_codes: list = None, sleep_sec: float = 0.3,
                                  stale_threshold_days: int = 5) -> int:
    """
    增量更新复权因子：只拉取数据库中缺失的最新部分

    优化策略：
    - 短路判断：若数据库中 max_date 距今 < stale_threshold_days 天，跳过该股票
      （不发 tushare 请求，不 sleep），避免对已最新股票做无效请求
    - sleep_sec 默认 0.3s：200积分账户 tushare 限制 200次/分钟（≈ 0.3s/次）
    - 限流自动退避：检测到"频率超限"时 sleep 60 秒再重试该股票
    - stale_threshold_days 默认 5：覆盖周末（周五→周一 diff=3）+ 1天缓冲

    Args:
        ts_codes: 股票代码列表，None 则自动获取全市场
        sleep_sec: 请求间隔（秒），避免触发tushare频率限制
        stale_threshold_days: 多少天未更新视为"陈旧"需要请求
    Returns:
        更新的记录数
    """
    from datasource.database import get_engine
    from tqdm import tqdm
    import re

    ensure_adj_factor_table()

    if ts_codes is None:
        with get_session() as session:
            from datasource.database import query_df
            pool_df = query_df(session, "stock_pools", columns=["ts_code"])
            ts_codes = pool_df["ts_code"].tolist()

    engine = get_engine()

    batch_size = 50
    total = 0
    skipped = 0
    requested = 0
    rate_limited = 0
    today = pd.Timestamp.now().normalize()
    today_str = today.strftime("%Y%m%d")

    # 限流错误模式：tushare 返回 "频率超限" 或 "rate limit"
    RATE_LIMIT_PATTERN = re.compile(r"频率超限|rate.?limit|访问频率")

    def _is_rate_limited(exc: Exception) -> bool:
        """检测是否触发 tushare 频率限制"""
        msg = str(exc)
        return bool(RATE_LIMIT_PATTERN.search(msg))

    print(f"开始增量更新复权因子: {len(ts_codes)} 只股票 (stale_threshold={stale_threshold_days}天, sleep={sleep_sec}s)")

    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i + batch_size]

        with engine.begin() as conn:
            placeholders = ", ".join([f"'{c}'" for c in batch])
            result = conn.execute(text(
                f"SELECT ts_code, MAX(trade_date) as max_date "
                f"FROM {ADJ_FACTOR_TABLE} "
                f"WHERE ts_code IN ({placeholders}) "
                f"GROUP BY ts_code"
            )).fetchall()

        max_date_map = {r[0]: pd.Timestamp(r[1]) for r in result if r[1]}

        for ts_code in tqdm(batch, desc=f"批次 {i//batch_size+1}/{(len(ts_codes)+batch_size-1)//batch_size}",
                            unit="只", leave=False):
            try:
                existing_max = max_date_map.get(ts_code)
                # 短路判断：若 max_date 距今 < threshold 天，跳过
                if existing_max is not None and (today - existing_max).days < stale_threshold_days:
                    skipped += 1
                    continue

                start = None
                if existing_max is not None:
                    start = (existing_max + pd.Timedelta(days=1)).strftime("%Y%m%d")
                # 额外短路：start > today 时（数据最新）跳过
                if start is not None and start > today_str:
                    skipped += 1
                    continue

                # 限流退避重试：最多 3 次
                for retry in range(3):
                    try:
                        cnt = fetch_and_save_adj_factor(ts_code, start_date=start, end_date=today_str)
                        total += cnt
                        requested += 1
                        break
                    except Exception as e:
                        if _is_rate_limited(e) and retry < 2:
                            rate_limited += 1
                            print(f"  ⚠️ 触发限流，sleep 60 秒后重试 ({ts_code})...")
                            time.sleep(60)
                        else:
                            raise
            except Exception as e:
                print(f"  {ts_code} 获取失败: {e}")
            time.sleep(sleep_sec)

    print(f"✅ 复权因子增量更新完成: 跳过 {skipped} 只 / 实际请求 {requested} 只 / 限流 {rate_limited} 次 / 新增 {total} 条记录")
    return total


def apply_adj_factor(df: pd.DataFrame, ts_code: str, freq: str = "d") -> pd.DataFrame:
    """
    对K线数据应用前复权转换

    前复权公式：前复权价格 = 不复权价格 × (历史adj_factor / 最新adj_factor)

    Args:
        df: K线数据 DataFrame，必须包含 open/high/low/close 列，index 为 bar_time
        ts_code: 股票代码，如 '688678.SH'
        freq: 周期（仅日线和周线需要复权）

    Returns:
        前复权后的 DataFrame（volume 不变，OHLC 调整）
    """
    if freq not in ("d", "w"):
        return df

    if df.empty:
        return df

    from datasource.database import get_engine

    ensure_adj_factor_table()

    engine = get_engine()

    with engine.begin() as conn:
        adj_df = pd.read_sql(
            text(f"SELECT trade_date, adj_factor FROM {ADJ_FACTOR_TABLE} "
                 f"WHERE ts_code = :ts_code ORDER BY trade_date"),
            conn,
            params={"ts_code": ts_code}
        )

    if adj_df.empty:
        return df

    adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])

    latest_adj = float(adj_df["adj_factor"].iloc[-1])
    if latest_adj == 0:
        return df

    adj_map = dict(zip(adj_df["trade_date"], adj_df["adj_factor"]))

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["_adj"] = df.index.normalize().map(adj_map)

    missing_adj = df["_adj"].isna()
    if missing_adj.any():
        adj_sorted = adj_df.set_index("trade_date")["adj_factor"].sort_index()
        for idx in df.index[missing_adj]:
            nearest = adj_sorted.index.get_indexer([idx.normalize()], method="ffill")[0]
            if nearest >= 0:
                df.loc[idx, "_adj"] = float(adj_sorted.iloc[nearest])

    df["_adj"] = df["_adj"].fillna(latest_adj)

    ratio = df["_adj"] / latest_adj

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col] * ratio

    df = df.drop(columns=["_adj"])

    return df


def apply_adj_factor_intraday(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """
    对分钟级K线数据应用前复权转换（按交易日映射adj_factor）

    分钟数据的 bar_time 为 datetime，通过 normalize() 映射到交易日，
    再查找对应日期的 adj_factor 进行前复权转换。
    同一交易日内的所有分钟bar使用相同的 adj_factor。

    前复权公式：前复权价格 = 不复权价格 × (历史adj_factor / 最新adj_factor)

    Args:
        df: 分钟级K线数据 DataFrame，必须包含 open/high/low/close 列，index 为 bar_time
        ts_code: 股票代码，如 '688678.SH'

    Returns:
        前复权后的 DataFrame（volume 不变，OHLC 调整）
    """
    if df.empty:
        return df

    from datasource.database import get_engine

    ensure_adj_factor_table()

    engine = get_engine()

    with engine.begin() as conn:
        adj_df = pd.read_sql(
            text(f"SELECT trade_date, adj_factor FROM {ADJ_FACTOR_TABLE} "
                 f"WHERE ts_code = :ts_code ORDER BY trade_date"),
            conn,
            params={"ts_code": ts_code}
        )

    if adj_df.empty:
        return df

    adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])

    latest_adj = float(adj_df["adj_factor"].iloc[-1])
    if latest_adj == 0:
        return df

    adj_map = dict(zip(adj_df["trade_date"], adj_df["adj_factor"]))

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["_adj"] = df.index.normalize().map(adj_map)

    missing_adj = df["_adj"].isna()
    if missing_adj.any():
        adj_sorted = adj_df.set_index("trade_date")["adj_factor"].sort_index()
        for idx in df.index[missing_adj]:
            nearest = adj_sorted.index.get_indexer([idx.normalize()], method="ffill")[0]
            if nearest >= 0:
                df.loc[idx, "_adj"] = float(adj_sorted.iloc[nearest])

    df["_adj"] = df["_adj"].fillna(latest_adj)

    ratio = df["_adj"] / latest_adj

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col] * ratio

    df = df.drop(columns=["_adj"])

    return df


def test_adj_factor(ts_code: str):
    """
    测试单只股票的前复权转换，对比数据库不复权数据和tushare前复权数据

    Args:
        ts_code: 股票代码，如 '688678.SH'
    """
    from tushare_data.config import TS_TOKEN
    import tushare as ts
    from datasource.database import get_engine

    ensure_adj_factor_table()

    engine = get_engine()

    with engine.begin() as conn:
        db_df = pd.read_sql(
            text("SELECT bar_time, open, high, low, close, volume "
                 "FROM stock_k_data "
                 "WHERE ts_code = :ts_code AND freq = 'd' "
                 "ORDER BY bar_time DESC LIMIT 30"),
            conn,
            params={"ts_code": ts_code}
        )

    if db_df.empty:
        print(f"数据库中无 {ts_code} 日线数据")
        return

    db_df["bar_time"] = pd.to_datetime(db_df["bar_time"])
    db_df = db_df.sort_values("bar_time").set_index("bar_time")

    qfq_df = apply_adj_factor(db_df, ts_code, freq="d")

    pro = ts.pro_api(TS_TOKEN)
    ts_df = ts.pro_bar(ts_code=ts_code, start_date="20260101", end_date="20260530", adj="qfq")

    if ts_df is not None and not ts_df.empty:
        ts_df["trade_date"] = pd.to_datetime(ts_df["trade_date"])
        ts_df = ts_df.set_index("trade_date").sort_index()

        common_idx = qfq_df.index.intersection(ts_df.index)
        if len(common_idx) > 0:
            print(f"\n{'='*80}")
            print(f"前复权验证: {ts_code}")
            print(f"{'='*80}")
            print(f"{'日期':<12} {'DB前复权close':>14} {'Tushare前复权close':>18} {'差异':>10}")
            print("-" * 58)
            for idx in common_idx[-10:]:
                db_close = qfq_df.loc[idx, "close"]
                ts_close = ts_df.loc[idx, "close"]
                diff = abs(db_close - ts_close)
                print(f"{str(idx)[:10]:<12} {db_close:>14.2f} {ts_close:>18.2f} {diff:>10.4f}")
        else:
            print("无公共日期可对比")
    else:
        print("tushare 前复权数据为空")

    print(f"\n数据库不复权最近5日:")
    print(db_df.tail(5)[["open", "high", "low", "close"]].to_string())
    print(f"\n前复权最近5日:")
    print(qfq_df.tail(5)[["open", "high", "low", "close"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复权因子管理工具")
    parser.add_argument("--ts_code", type=str, help="单只股票代码，如 688678.SH")
    parser.add_argument("--incremental", action="store_true", help="增量更新")
    parser.add_argument("--test", type=str, metavar="TS_CODE", help="测试单只股票前复权验证")
    parser.add_argument("--start_date", type=str, help="起始日期，如 20200101")
    parser.add_argument("--end_date", type=str, help="结束日期，如 20260530")
    args = parser.parse_args()

    if args.test:
        test_adj_factor(args.test)
    elif args.incremental:
        ts_codes = None
        if args.ts_code:
            ts_codes = [args.ts_code]
        cnt = incremental_update_adj_factor(ts_codes)
        print(f"增量更新完成，共 {cnt} 条记录")
    elif args.ts_code:
        cnt = fetch_and_save_adj_factor(args.ts_code, args.start_date, args.end_date)
        print(f"保存 {cnt} 条记录")
    else:
        from datasource.database import query_df
        with get_session() as session:
            pool_df = query_df(session, "stock_pools", columns=["ts_code"])
        ts_codes = pool_df["ts_code"].tolist()
        print(f"全量回补 {len(ts_codes)} 只股票的复权因子...")
        cnt = batch_fetch_adj_factor(ts_codes, args.start_date, args.end_date)
        print(f"全量回补完成，共 {cnt} 条记录")
