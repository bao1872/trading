# -*- coding: utf-8 -*-
"""
交易日历工具模块（SSOT）

功能：提供权威的交易日判断函数，供调度器和其他模块统一调用
数据源优先级：
    1. tushare trade_cal API（含完整节假日信息）
    2. 数据库 stock_k_data 表（降级）
    3. weekday 判断（最后降级，记录 WARNING）

用法：
    from datasource.trade_calendar import is_trading_day

    # 检查今天是否为交易日
    if is_trading_day():
        print("今天是交易日")

    # 检查指定日期
    if is_trading_day("2026-04-04"):
        print("清明节是交易日")

示例：
    python -c "from datasource.trade_calendar import is_trading_day; print(is_trading_day())"
    python -c "from datasource.trade_calendar import is_trading_day; print(is_trading_day('2026-04-04'))"

副作用：无（只读查询，不写库/不改表/不写文件）
"""

import logging
from datetime import date, datetime
from typing import Optional, Union

logger = logging.getLogger(__name__)

_cache_date: Optional[str] = None
_cache_result: Optional[bool] = None


def _parse_date(target_date: Optional[Union[str, date, datetime]] = None) -> date:
    """将输入解析为 date 对象"""
    if target_date is None:
        return date.today()
    if isinstance(target_date, datetime):
        return target_date.date()
    if isinstance(target_date, date):
        return target_date
    if isinstance(target_date, str):
        return datetime.strptime(target_date, "%Y-%m-%d").date()
    raise ValueError(f"不支持的日期类型：{type(target_date)}")


def _check_tushare(target: date) -> Optional[bool]:
    """通过 tushare trade_cal API 检查是否为交易日"""
    try:
        import tushare as ts
        from tushare_data.config import TS_TOKEN

        pro = ts.pro_api(TS_TOKEN)
        date_str = target.strftime("%Y%m")
        df = pro.trade_cal(exchange="SSE", start_date=date_str + "01")

        if df is not None and not df.empty:
            target_str = target.strftime("%Y%m%d")
            row = df[df["cal_date"] == target_str]
            if not row.empty:
                return bool(row.iloc[0]["is_open"])
    except Exception as e:
        logger.warning(f"tushare trade_cal 查询失败，将降级到数据库查询：{e}")
    return None


def _check_database(target: date) -> Optional[bool]:
    """通过数据库 stock_k_data 表检查是否为交易日"""
    try:
        from datasource.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            sql = text(
                "SELECT 1 FROM stock_k_data "
                "WHERE freq = 'd' AND DATE(bar_time) = :target_date "
                "LIMIT 1"
            )
            result = conn.execute(sql, {"target_date": target}).fetchone()
            return result is not None
    except Exception as e:
        logger.warning(f"数据库交易日查询失败，将降级到 weekday 判断：{e}")
    return None


def _check_weekday(target: date) -> bool:
    """仅通过 weekday 判断（不考虑节假日）"""
    is_weekday = target.weekday() < 5
    if is_weekday:
        logger.warning(
            f"交易日判断降级为 weekday 模式，{target} 可能是节假日但被误判为交易日"
        )
    return is_weekday


def is_trading_day(target_date: Optional[Union[str, date, datetime]] = None) -> bool:
    """检查指定日期是否为交易日

    Args:
        target_date: 日期对象、字符串(YYYY-MM-DD)或 datetime，默认今天

    Returns:
        True 表示交易日，False 表示非交易日

    数据源优先级：
        1. tushare trade_cal API（含节假日，缓存当日结果）
        2. 数据库 stock_k_data 表（降级）
        3. weekday 判断（最后降级，记录 WARNING）
    """
    global _cache_date, _cache_result

    target = _parse_date(target_date)
    target_str = target.strftime("%Y-%m-%d")

    if target_str == _cache_date and _cache_result is not None:
        return _cache_result

    result = _check_tushare(target)
    if result is None:
        result = _check_database(target)
    if result is None:
        result = _check_weekday(target)

    if target_str == date.today().strftime("%Y-%m-%d"):
        _cache_date = target_str
        _cache_result = result

    logger.info(f"交易日检查：{target_str} -> {'交易日' if result else '非交易日'}")
    return result


if __name__ == "__main__":
    test_dates = [
        None,
        "2026-04-03",
        "2026-04-04",
        "2026-04-05",
        "2026-04-06",
        "2026-04-07",
        "2026-05-01",
        "2026-05-05",
    ]
    for d in test_dates:
        label = d if d else "今天"
        print(f"{label}: {'交易日' if is_trading_day(d) else '非交易日'}")
