# -*- coding: utf-8 -*-
"""
因子/事件统一读取接口

Purpose: 所有脚本统一使用此接口从 factor_value / event_trigger 读取因子和事件数据
How to Run: 无需单独运行，被其他脚本 import

Examples:
    from datasource.factor_reader import load_factors_batch
    factors = load_factors_batch(["000001", "000002"], "2026-04-30", freq="1d", lookback_days=5)

    from datasource.factor_reader import load_factors_from_db
    df = load_factors_from_db("000001.SZ", "2022-01-01", "2026-04-30", freq="1w")
"""
import pandas as pd
from sqlalchemy import text


def _get_engine():
    from datasource.database import get_engine
    return get_engine()


def _code_to_db(code: str) -> str:
    """将纯数字代码转为数据库格式 000001 -> 000001.SZ"""
    if "." in code:
        return code
    if code.startswith(("6", "5")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _db_to_code(db_code: str) -> str:
    """将数据库格式转为纯数字代码 000001.SZ -> 000001"""
    return db_code.split(".")[0]


def _get_factor_table(freq: str, use_new_table: bool) -> str:
    """根据频率和是否使用新表获取因子表名"""
    if use_new_table:
        return f"factor_value_{freq}"
    return "factor_value"


def _get_event_table(freq: str, use_new_table: bool) -> str:
    """根据频率和是否使用新表获取事件表名"""
    if use_new_table:
        return f"event_trigger_{freq}"
    return "event_trigger"


def load_factors_from_db(ts_code: str, start_date: str, end_date: str,
                         freq: str = "1d", factor_names: list = None,
                         use_new_table: bool = True) -> pd.DataFrame:
    """从 factor_value 读取单只股票的因子（返回宽表 DataFrame）

    Args:
        ts_code: 股票代码（纯数字如 '000001' 或带后缀如 '000001.SZ'）
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        freq: 频率 '1d' / '1w'
        factor_names: 指定因子名列表，None=全部
        use_new_table: 是否读取新表（默认 False，读取旧表）

    Returns:
        DataFrame: index=as_of_date, columns=factor_name, 宽表格式
    """
    db_code = _code_to_db(ts_code)
    engine = _get_engine()
    table_name = _get_factor_table(freq, use_new_table)

    if factor_names:
        placeholders = ", ".join([f"'{n}'" for n in factor_names])
        name_filter = f"AND factor_name IN ({placeholders})"
    else:
        name_filter = ""

    if use_new_table:
        sql = text(f"""
            SELECT as_of_date, factor_name, factor_value
            FROM {table_name}
            WHERE ts_code = :code
              AND as_of_date >= :start AND as_of_date <= :end
              {name_filter}
            ORDER BY as_of_date, factor_name
        """)
        params = {"code": db_code, "start": start_date, "end": end_date}
    else:
        sql = text(f"""
            SELECT as_of_date, factor_name, factor_value
            FROM {table_name}
            WHERE ts_code = :code AND freq = :freq
              AND as_of_date >= :start AND as_of_date <= :end
              {name_filter}
            ORDER BY as_of_date, factor_name
        """)
        params = {"code": db_code, "freq": freq, "start": start_date, "end": end_date}

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    wide = df.pivot(index="as_of_date", columns="factor_name", values="factor_value")
    wide.columns.name = None
    return wide


def load_events_from_db(ts_code: str, start_date: str, end_date: str,
                        freq: str = "1d", event_names: list = None,
                        use_new_table: bool = True) -> pd.DataFrame:
    """从 event_trigger 读取单只股票的事件（返回宽表 DataFrame）

    Args:
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率
        event_names: 指定事件名列表，None=全部
        use_new_table: 是否读取新表（默认 False，读取旧表）

    Returns:
        DataFrame: index=as_of_date, columns=event_name (triggered bool)
    """
    db_code = _code_to_db(ts_code)
    engine = _get_engine()
    table_name = _get_event_table(freq, use_new_table)

    if event_names:
        placeholders = ", ".join([f"'{n}'" for n in event_names])
        name_filter = f"AND event_name IN ({placeholders})"
    else:
        name_filter = ""

    if use_new_table:
        sql = text(f"""
            SELECT as_of_date, event_name, triggered
            FROM {table_name}
            WHERE ts_code = :code
              AND as_of_date >= :start AND as_of_date <= :end
              {name_filter}
            ORDER BY as_of_date, event_name
        """)
        params = {"code": db_code, "start": start_date, "end": end_date}
    else:
        sql = text(f"""
            SELECT as_of_date, event_name, triggered
            FROM {table_name}
            WHERE ts_code = :code AND freq = :freq
              AND as_of_date >= :start AND as_of_date <= :end
              {name_filter}
            ORDER BY as_of_date, event_name
        """)
        params = {"code": db_code, "freq": freq, "start": start_date, "end": end_date}

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    wide = df.pivot(index="as_of_date", columns="event_name", values="triggered")
    wide.columns.name = None
    return wide


def load_factors_batch(codes: list, target_date: str, freq: str = "1d",
                       lookback_days: int = 1, factor_names: list = None,
                       use_new_table: bool = True) -> dict:
    """批量读取多只股票的最近N天因子（返回 {code: wide_df} 字典）

    Args:
        codes: 股票代码列表（纯数字）
        target_date: 目标日期 YYYY-MM-DD
        freq: 频率
        lookback_days: 回溯天数（从 target_date 往前推，含当日）
        factor_names: 指定因子名列表，None=全部
        use_new_table: 是否读取新表（默认 False，读取旧表）

    Returns:
        dict: {纯数字代码: DataFrame(as_of_date 为 index, factor_name 为 columns)}
    """
    if not codes:
        return {}

    db_codes = [_code_to_db(c) for c in codes]
    engine = _get_engine()
    table_name = _get_factor_table(freq, use_new_table)

    if factor_names:
        name_placeholders = ", ".join([f"'{n}'" for n in factor_names])
        name_filter = f"AND factor_name IN ({name_placeholders})"
    else:
        name_filter = ""

    code_placeholders = ", ".join([f"'{c}'" for c in db_codes])

    import datetime
    target_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    start_dt = target_dt - datetime.timedelta(days=lookback_days - 1)
    start_str = start_dt.strftime("%Y-%m-%d")

    if use_new_table:
        sql = text(f"""
            SELECT ts_code, factor_name, factor_value, as_of_date
            FROM {table_name}
            WHERE ts_code IN ({code_placeholders})
              AND as_of_date >= :start
              AND as_of_date <= :target
              {name_filter}
            ORDER BY ts_code, as_of_date, factor_name
        """)
        params = {"start": start_str, "target": target_date}
    else:
        sql = text(f"""
            SELECT ts_code, factor_name, factor_value, as_of_date
            FROM {table_name}
            WHERE ts_code IN ({code_placeholders}) AND freq = :freq
              AND as_of_date >= :start
              AND as_of_date <= :target
              {name_filter}
            ORDER BY ts_code, as_of_date, factor_name
        """)
        params = {"freq": freq, "start": start_str, "target": target_date}

    with engine.connect() as conn:
        factor_long = pd.read_sql(sql, conn, params=params)

    if factor_long.empty:
        return {}

    factor_long["raw_code"] = factor_long["ts_code"].apply(_db_to_code)

    result = {}
    for raw_code, grp in factor_long.groupby("raw_code"):
        wide = grp.pivot(index="as_of_date", columns="factor_name", values="factor_value")
        wide.columns.name = None
        result[raw_code] = wide
    return result


def load_factors_last(codes: list, target_date: str, freq: str = "1d",
                      factor_names: list = None, use_new_table: bool = True) -> pd.DataFrame:
    """读取多只股票指定日期的因子快照（宽表，一行一股）

    Args:
        codes: 股票代码列表（纯数字）
        target_date: 目标日期
        freq: 频率
        factor_names: 指定因子名列表
        use_new_table: 是否读取新表（默认 False，读取旧表）

    Returns:
        DataFrame: 一行一股，columns = ts_code + 因子列名
    """
    if not codes:
        return pd.DataFrame()

    db_codes = [_code_to_db(c) for c in codes]
    engine = _get_engine()
    table_name = _get_factor_table(freq, use_new_table)

    if factor_names:
        name_placeholders = ", ".join([f"'{n}'" for n in factor_names])
        name_filter = f"AND factor_name IN ({name_placeholders})"
    else:
        name_filter = ""

    code_placeholders = ", ".join([f"'{c}'" for c in db_codes])

    if use_new_table:
        sql = text(f"""
            SELECT ts_code, factor_name, factor_value
            FROM {table_name}
            WHERE ts_code IN ({code_placeholders})
              AND as_of_date = :target
              {name_filter}
            ORDER BY ts_code, factor_name
        """)
        params = {"target": target_date}
    else:
        sql = text(f"""
            SELECT ts_code, factor_name, factor_value
            FROM {table_name}
            WHERE ts_code IN ({code_placeholders}) AND freq = :freq
              AND as_of_date = :target
              {name_filter}
            ORDER BY ts_code, factor_name
        """)
        params = {"freq": freq, "target": target_date}

    with engine.connect() as conn:
        factor_long = pd.read_sql(sql, conn, params=params)

    if factor_long.empty:
        return pd.DataFrame()

    factor_long["raw_code"] = factor_long["ts_code"].apply(_db_to_code)
    wide = factor_long.pivot(index="raw_code", columns="factor_name", values="factor_value")
    wide.columns.name = None
    wide = wide.reset_index().rename(columns={"raw_code": "ts_code"})
    return wide


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("测试 factor_reader 接口...")
    codes = ["000001", "000002"]
    result = load_factors_last(codes, "2026-04-30", freq="1d",
                               factor_names=["dsa_dir", "bbmacd", "vol_zscore_10"])
    print(f"load_factors_last: {len(result)} 行, {len(result.columns)} 列")
    print(result.head())

    batch = load_factors_batch(codes, "2026-04-30", freq="1d", lookback_days=3,
                               factor_names=["dsa_dir", "bbmacd"])
    for code, df in batch.items():
        print(f"  {code}: {len(df)} 行 x {len(df.columns)} 列")

    # 测试新表读取
    result_new = load_factors_last(codes, "2026-04-30", freq="1d",
                                   factor_names=["dsa_dir", "bbmacd", "vol_zscore_10"],
                                   use_new_table=True)
    print(f"load_factors_last (new table): {len(result_new)} 行, {len(result_new.columns)} 列")
    print("✅ 接口测试通过")
