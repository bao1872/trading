# -*- coding: utf-8 -*-
"""
K线数据加载模块 - 从 PostgreSQL 数据库统一加载K线数据

支持多周期：5m, 15m, 60m, d, w
支持格式：dict {code: {name, data}} 用于 backtrader，DataFrame 用于其他场景

Usage:
    from datasource.k_data_loader import load_k_data, load_k_data_as_dict

    # 返回 dict 格式（backtrader 用）
    data_dict = load_k_data(freq='d')

    # 返回 DataFrame 格式
    df = load_k_data('600489.SH', freq='d')
"""
import os
import sys
from typing import Dict, List, Optional, Union

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df


FREQ_MAP = {
    'd': 'd',
    'daily': 'd',
    'day': 'd',
    'w': 'w',
    'weekly': 'w',
    'week': 'w',
    'm': 'm',
    'monthly': 'm',
    'month': 'm',
    '60m': '60m',
    '60min': '60m',
    '1h': '60m',
    'hour': '60m',
    '15m': '15m',
    '15min': '15m',
    '5m': '5m',
    '5min': '5m',
}


def load_k_data(
    ts_code: Optional[str] = None,
    freq: str = 'd',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> Union[pd.DataFrame, Dict[str, Dict]]:
    """
    加载K线数据

    Args:
        ts_code: 股票代码，如 "600489.SH"，None 表示加载所有股票
        freq: 周期
        start_date: 开始日期
        end_date: 结束日期
        limit: 返回记录数限制

    Returns:
        单只股票返回 DataFrame，多只股票返回 dict {code: {name, data}}
    """
    db_freq = FREQ_MAP.get(freq.lower())
    if not db_freq:
        raise ValueError(f"不支持的频率: {freq}")

    filters = {"freq": db_freq}
    if ts_code:
        filters["ts_code"] = ts_code
    if start_date:
        filters["bar_time >= "] = start_date
    if end_date:
        filters["bar_time <= "] = end_date

    with get_session() as session:
        df = query_df(
            session,
            table_name="stock_k_data",
            filters=filters,
            order_by="+ts_code,+bar_time",
            limit=limit
        )

    if df.empty:
        return pd.DataFrame() if ts_code else {}

    df["bar_time"] = pd.to_datetime(df["bar_time"])
    df = df.sort_values(["ts_code", "bar_time"])

    if ts_code:
        df = df.set_index("bar_time")
        return df

    result = {}
    for code, group in df.groupby("ts_code"):
        group = group.set_index("bar_time")
        group = group.sort_index()
        result[code] = {
            "name": code,
            "data": group
        }

    return result


def iter_k_data(
    freq: str = 'd',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    迭代器方式逐股票加载K线数据（内存友好）
    一次查询所有数据到DataFrame，按股票分组逐个yield，内存占用低

    Args:
        freq: 周期 ('daily', 'weekly', 'min60', 'min15')
        start_date: 开始日期
        end_date: 结束日期

    Yields:
        (code, name, DataFrame) 元组
    """
    freq_to_db = {
        'daily': 'd',
        'weekly': 'w',
        'min60': '60m',
        'min15': '15m',
    }
    db_freq = freq_to_db.get(freq, freq)

    filters = {"freq": db_freq}
    if start_date:
        filters["bar_time >= "] = start_date
    if end_date:
        filters["bar_time <= "] = end_date

    with get_session() as session:
        df = query_df(
            session,
            table_name="stock_k_data",
            filters=filters,
            order_by="+ts_code,+bar_time"
        )

    if df.empty:
        return

    df["bar_time"] = pd.to_datetime(df["bar_time"])
    codes = df['ts_code'].unique()
    name_map = build_name_map(list(codes))

    db_columns = ["id", "ts_code", "freq", "created_at"]
    for code in codes:
        stock_df = df[df['ts_code'] == code].copy()
        stock_df = stock_df.drop(columns=[c for c in db_columns if c in stock_df.columns], errors="ignore")
        stock_df = stock_df.set_index("bar_time")
        yield code, name_map.get(code, code), stock_df


def iter_k_data_with_names(
    freq: str = 'd',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name_map: Optional[dict] = None,
    limit_stocks: Optional[int] = None
):
    """
    迭代器方式逐股票加载K线数据（外部传入name_map，避免重复查询）
    一次查询所有数据到DataFrame，按股票分组逐个yield，内存占用低

    Args:
        freq: 周期
        start_date: 开始日期
        end_date: 结束日期
        name_map: 股票代码->名称的映射，None则自动从数据库加载
        limit_stocks: 限制股票数量（用于测试），None表示不限制

    Yields:
        (code, name, DataFrame) 元组
    """
    freq_to_db = {
        'daily': 'd',
        'weekly': 'w',
        'min60': '60m',
        'min15': '15m',
    }
    db_freq = freq_to_db.get(freq, freq)

    filters = {"freq": db_freq}
    if start_date:
        filters["bar_time >= "] = start_date
    if end_date:
        filters["bar_time <= "] = end_date

    with get_session() as session:
        df = query_df(
            session,
            table_name="stock_k_data",
            filters=filters,
            order_by="+ts_code,+bar_time"
        )

    if df.empty:
        return

    df["bar_time"] = pd.to_datetime(df["bar_time"])
    codes = df['ts_code'].unique()

    if limit_stocks:
        codes = codes[:limit_stocks]

    if name_map is None:
        name_map = build_name_map(list(codes))

    db_columns = ["id", "ts_code", "freq", "created_at"]
    for code in codes:
        stock_df = df[df['ts_code'] == code].copy()
        stock_df = stock_df.drop(columns=[c for c in db_columns if c in stock_df.columns], errors="ignore")
        stock_df = stock_df.set_index("bar_time")
        yield code, name_map.get(code, code), stock_df


def load_k_data_as_dict(
    freq: str = 'd',
    limit_stocks: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Dict]:
    """
    加载K线数据为 dict 格式（兼容 backtrader 旧代码）

    Args:
        freq: 周期 ('daily', 'weekly', 'min60', 'min15')
        limit_stocks: 限制股票数量（用于测试）
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        dict {code: {'name': str, 'data': DataFrame}}
    """
    freq_to_db = {
        'daily': 'd',
        'weekly': 'w',
        'min60': '60m',
        'min15': '15m',
    }
    db_freq = freq_to_db.get(freq, freq)

    filters = {"freq": db_freq}
    if start_date:
        filters["bar_time >= "] = start_date
    if end_date:
        filters["bar_time <= "] = end_date

    with get_session() as session:
        df = query_df(
            session,
            table_name="stock_k_data",
            filters=filters,
            order_by="+ts_code,+bar_time"
        )

    if df.empty:
        return {}

    df["bar_time"] = pd.to_datetime(df["bar_time"])
    df = df.sort_values(["ts_code", "bar_time"])

    result = {}
    codes = df['ts_code'].unique()
    if limit_stocks:
        codes = codes[:limit_stocks]

    code_list = list(codes)
    with get_session() as session:
        name_df = query_df(
            session,
            table_name="stock_pools",
            columns=["ts_code", "name"],
            filters={"ts_code": code_list} if code_list else {}
        )
    name_map = dict(zip(name_df['ts_code'], name_df['name'])) if not name_df.empty else {}

    db_columns = ["id", "ts_code", "freq", "created_at"]
    for code in codes:
        stock_df = df[df['ts_code'] == code].copy()
        stock_df = stock_df.drop(columns=[c for c in db_columns if c in stock_df.columns], errors="ignore")
        stock_df = stock_df.set_index("bar_time")
        stock_df = stock_df.sort_index()
        result[code] = {
            "name": name_map.get(code, code),
            "data": stock_df
        }

    return result


def load_all_stock_data(
    freq: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit_stocks: Optional[int] = None
) -> tuple:
    """
    一次性加载所有股票数据到DataFrame（统一实现，SSOT）

    Args:
        freq: 周期 ('daily', 'weekly', 'min60', 'min15')
        start_date: 开始日期
        end_date: 结束日期
        limit_stocks: 限制股票数量（用于测试），None表示不限制

    Returns:
        (DataFrame, name_map) 元组
    """
    df_list = []
    name_map = {}

    for code, name, df in iter_k_data_with_names(
        freq=freq, start_date=start_date, end_date=end_date,
        name_map=None, limit_stocks=limit_stocks
    ):
        if df.empty or 'volume' not in df.columns:
            continue
        df = df.reset_index()
        df['ts_code'] = code
        df['name'] = name
        df_list.append(df)
        name_map[code] = name

    if not df_list:
        return pd.DataFrame(), name_map

    result = pd.concat(df_list, ignore_index=False)
    result = result.sort_values(['ts_code', 'bar_time'])
    result = result.reset_index(drop=True)
    return result, name_map


def get_stock_name(ts_code: str) -> Optional[str]:
    """获取股票名称"""
    with get_session() as session:
        result = query_df(
            session,
            table_name="stock_pools",
            columns=["name"],
            filters={"ts_code": ts_code}
        )
    if result.empty:
        return None
    return result.iloc[0]["name"]


def build_name_map(codes: Optional[List[str]] = None) -> Dict[str, str]:
    """
    构建股票代码到名称的映射字典

    Args:
        codes: 股票代码列表，None 表示加载所有股票

    Returns:
        dict {ts_code: name}
    """
    filters = {}
    if codes:
        filters["ts_code"] = codes

    with get_session() as session:
        name_df = query_df(
            session,
            table_name="stock_pools",
            columns=["ts_code", "name"],
            filters=filters
        )

    if name_df.empty:
        return {}

    return dict(zip(name_df['ts_code'], name_df['name']))


def get_all_codes(freq: str = 'd') -> List[str]:
    """获取指定频率的所有股票代码"""
    db_freq = FREQ_MAP.get(freq.lower())
    if not db_freq:
        return []

    with get_session() as session:
        df = query_df(
            session,
            table_name="stock_k_data",
            columns=["ts_code"],
            filters={"freq": db_freq}
        )

    if df.empty:
        return []
    return df["ts_code"].unique().tolist()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="K线数据加载测试")
    parser.add_argument("--freq", type=str, default="d", help="周期")
    parser.add_argument("--code", type=str, default=None, help="股票代码")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量")
    parser.add_argument("--dict", action="store_true", help="返回 dict 格式")
    args = parser.parse_args()

    if args.code:
        df = load_k_data(args.code, freq=args.freq)
        print(f"股票: {args.code}, 频率: {args.freq}, 记录数: {len(df)}")
        if not df.empty:
            print(df.head())
    elif args.dict:
        data = load_k_data_as_dict(freq=args.freq, limit_stocks=args.limit)
        print(f"股票数量: {len(data)}")
        if data:
            sample_code = list(data.keys())[0]
            print(f"示例 {sample_code}: {len(data[sample_code]['data'])} bars")
    else:
        codes = get_all_codes(args.freq)
        print(f"数据库中 {args.freq} 频率的股票数量: {len(codes)}")