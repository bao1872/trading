# -*- coding: utf-8 -*-
"""
Purpose: 验证 pytdx 能否获取集合竞价相关数据（1分钟K线9:25/9:30、逐笔成交9:25、分时数据）
Inputs:  股票代码 000001，日期 20260512
Outputs: 打印各类数据中是否包含集合竞价时间段记录
How to Run: python scripts/verify_auction_data.py
Examples:
    python scripts/verify_auction_data.py
    python scripts/verify_auction_data.py --date 20260511
Side Effects: 仅读取数据，不写库/改表/写文件
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from datasource.pytdx_client import connect_pytdx, market_from_code

SYMBOL = "000001"
MARKET = market_from_code(SYMBOL)


def test_1min_kline(api, date_int):
    """测试1分钟K线是否包含9:25/9:30的bar"""
    print("=" * 70)
    print(f"【测试1】1分钟K线数据 (category=8)")
    print("=" * 70)

    all_bars = []
    start = 0
    fetch_count = 800
    while len(all_bars) < 8000:
        data = api.get_security_bars(8, MARKET, SYMBOL, start, fetch_count)
        if not data:
            break
        all_bars.extend(data)
        if len(data) < fetch_count:
            break
        start += fetch_count

    if not all_bars:
        print("  ❌ 无数据返回")
        return

    df = pd.DataFrame(all_bars)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    elif {"year", "month", "day", "hour", "minute"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour", "minute"]].astype(int)
        ).dt.tz_localize(None)

    df = df.sort_values("datetime", ascending=True).reset_index(drop=True)

    date_str = str(date_int)
    target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    day_df = df[df["datetime"].dt.strftime("%Y-%m-%d") == target_date]

    print(f"  总记录数: {len(df)}")
    print(f"  日期范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  目标日期 {target_date} 的记录数: {len(day_df)}")

    if day_df.empty:
        print("  ❌ 目标日期无1分钟K线数据")
        return

    print(f"\n  目标日期前5条K线:")
    for _, row in day_df.head(5).iterrows():
        print(f"    {row['datetime']} | O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} V={row['vol']}")

    time_strs = day_df["datetime"].dt.strftime("%H:%M").tolist()
    has_925 = "09:25" in time_strs
    has_930 = "09:30" in time_strs
    print(f"\n  是否包含 09:25 K线: {'✅ 是' if has_925 else '❌ 否'}")
    print(f"  是否包含 09:30 K线: {'✅ 是' if has_930 else '❌ 否'}")
    print(f"  最早K线时间: {day_df['datetime'].min()}")
    print(f"  该日K线时间段: {time_strs[:10]}{'...' if len(time_strs) > 10 else ''}")


def test_tick_data(api, date_int):
    """测试逐笔成交数据是否包含9:25的记录（集合竞价撮合）"""
    print("\n" + "=" * 70)
    print(f"【测试2】逐笔成交数据 (get_history_transaction_data)")
    print("=" * 70)

    all_ticks = []
    start = 0
    page_size = 2000
    while True:
        data = api.get_history_transaction_data(MARKET, SYMBOL, start, page_size, date_int)
        if not data:
            break
        all_ticks.extend(data)
        if len(data) < page_size:
            break
        start += page_size
        if start > 50000:
            break

    if not all_ticks:
        print("  ❌ 无数据返回")
        return

    df = pd.DataFrame(all_ticks).drop_duplicates()
    print(f"  总记录数: {len(df)}")
    print(f"  列名: {list(df.columns)}")

    if "time" in df.columns:
        time_col = df["time"].astype(str)
        has_925 = time_col.str.startswith("09:25").any()
        has_930 = time_col.str.startswith("09:30").any()

        print(f"\n  是否包含 09:25 开头的记录: {'✅ 是' if has_925 else '❌ 否'}")
        print(f"  是否包含 09:30 开头的记录: {'✅ 是' if has_930 else '❌ 否'}")

        if has_925:
            auction_ticks = df[time_col.str.startswith("09:25")]
            print(f"  09:25 记录数: {len(auction_ticks)}")
            print(f"\n  09:25 前5条:")
            for _, row in auction_ticks.head(5).iterrows():
                print(f"    {row.get('time', '')} | 价格={row.get('price', '')} | 成交量={row.get('vol', '')} | 买卖={row.get('buyorsell', '')}")

        print(f"\n  最早5条记录:")
        for _, row in df.head(5).iterrows():
            print(f"    {row.get('time', '')} | 价格={row.get('price', '')} | 成交量={row.get('vol', '')} | 买卖={row.get('buyorsell', '')}")

        unique_times = sorted(time_col.str[:5].unique())
        print(f"\n  所有时间前缀(分钟): {unique_times[:15]}{'...' if len(unique_times) > 15 else ''}")
    else:
        print(f"  ⚠️ 无 time 列，可用列: {list(df.columns)}")
        print(f"  前3行:\n{df.head(3).to_string()}")


def test_minute_time_data(api, date_int):
    """测试分时数据 (get_history_minute_time_data)"""
    print("\n" + "=" * 70)
    print(f"【测试3】分时数据 (get_history_minute_time_data)")
    print("=" * 70)

    data = api.get_history_minute_time_data(MARKET, SYMBOL, date_int)

    if not data:
        print("  ❌ 无数据返回")
        return

    df = pd.DataFrame(data)
    print(f"  总记录数: {len(df)}")
    print(f"  列名: {list(df.columns)}")

    print(f"\n  前5条:")
    for _, row in df.head(5).iterrows():
        print(f"    {row.to_dict()}")

    print(f"\n  最后5条:")
    for _, row in df.tail(5).iterrows():
        print(f"    {row.to_dict()}")

    if "time" in df.columns:
        time_col = df["time"].astype(str)
        unique_times = sorted(time_col.unique())
        print(f"\n  时间点数量: {len(unique_times)}")
        print(f"  最早时间: {unique_times[0] if unique_times else 'N/A'}")
        print(f"  最晚时间: {unique_times[-1] if unique_times else 'N/A'}")
        has_925 = any(t.startswith("09:25") for t in unique_times)
        has_930 = any(t.startswith("09:30") for t in unique_times)
        print(f"  是否包含 09:25: {'✅ 是' if has_925 else '❌ 否'}")
        print(f"  是否包含 09:30: {'✅ 是' if has_930 else '❌ 否'}")


def test_minute_time_data_raw(api, date_int):
    """直接打印 get_history_minute_time_data 的原始返回"""
    print("\n" + "=" * 70)
    print(f"【测试4】分时数据原始返回 (前3条)")
    print("=" * 70)

    data = api.get_history_minute_time_data(MARKET, SYMBOL, date_int)
    if not data:
        print("  ❌ 无数据")
        return

    for i, item in enumerate(data[:3]):
        print(f"  [{i}] type={type(item)} value={item}")


def main():
    parser = argparse.ArgumentParser(description="验证pytdx集合竞价数据")
    parser.add_argument("--date", type=int, default=20260512, help="日期整数，如20260512")
    args = parser.parse_args()

    date_int = args.date
    print(f"验证股票: {SYMBOL} (市场={MARKET})")
    print(f"目标日期: {date_int}")

    api = connect_pytdx()
    try:
        test_1min_kline(api, date_int)
        test_tick_data(api, date_int)
        test_minute_time_data(api, date_int)
        test_minute_time_data_raw(api, date_int)

        print("\n" + "=" * 70)
        print("【总结】")
        print("=" * 70)
        print("1分钟K线: 检查是否有 09:25/09:30 的bar（集合竞价K线）")
        print("逐笔成交: 检查是否有 09:25 的记录（集合竞价撮合成交）")
        print("分时数据: 检查是否包含集合竞价时间段的数据点")
    finally:
        api.disconnect()
        print("\n连接已断开")


if __name__ == "__main__":
    main()
