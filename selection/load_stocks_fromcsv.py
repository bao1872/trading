# -*- coding: utf-8 -*-
"""
从CSV导入精选股票池数据到数据库并计算逐笔统计

Purpose: 读取 /root/trading/selection/ 目录下的CSV文件，导入到 stock_selected_picks 表，并计算最近5天逐笔成交额统计
Inputs:
    - CSV文件：/root/trading/selection/ 目录下唯一的CSV文件
    - CSV字段：代码,名称,投资推荐分,推荐档位,操作建议,主题标签,估值判断,定位,持续性,业绩趋势,上涨空间%,
               周线Z分,周线VolZ,日线VolZ,VWAP偏离%,排序理由,板块,炒作逻辑,风险因素
Outputs:
    - 数据库表 stock_selected_picks 写入数据
    - 更新 avg_tick_amount（平均每笔成交额）、max_tick_amount（最大成交额）、min_tick_amount（最小成交额）
How to Run:
    python selection/load_stocks_fromcsv.py
Examples:
    python selection/load_stocks_fromcsv.py
    python selection/load_stocks_fromcsv.py --help
Side Effects:
    - 向 stock_selected_picks 表写入数据（使用最近交易日作为pick_date）
    - 会删除同一日期的旧数据后重新插入（幂等性设计）
    - 会查询 stock_selection_results 表补充 signal_date（事件日期）
    - 通过 pytdx 获取最近5个交易日逐笔成交数据并计算统计指标
"""
import glob
import os
import sys
import traceback

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session
from datasource.pytdx_client import connect_pytdx, market_from_code


SELECTION_DIR = "/root/trading/selection"

TICK_STATS_COLS = [
    ("avg_tick_amount", "DECIMAL(10,2)", "平均每笔成交额(万元)"),
    ("max_tick_amount", "DECIMAL(10,2)", "最大成交额(万元)"),
    ("min_tick_amount", "DECIMAL(10,2)", "最小成交额(万元)"),
]

NEW_PICKS_COLS = [
    ("recommend_score", "DECIMAL(5,2)", "投资推荐分"),
    ("recommend_grade", "VARCHAR(50)", "推荐档位"),
    ("action_advice", "VARCHAR(100)", "操作建议"),
    ("theme_tags", "VARCHAR(200)", "主题标签"),
    ("valuation_judge", "VARCHAR(100)", "估值判断"),
    ("positioning", "VARCHAR(50)", "定位"),
    ("sustainability", "VARCHAR(20)", "持续性"),
    ("profit_trend", "VARCHAR(50)", "业绩趋势"),
    ("upside_pct", "DECIMAL(6,2)", "上涨空间%"),
    ("weekly_z", "DECIMAL(8,4)", "周线Z分"),
    ("weekly_vol_z", "DECIMAL(8,4)", "周线VolZ"),
    ("daily_vol_z", "DECIMAL(8,4)", "日线VolZ"),
    ("vwap_deviation", "DECIMAL(8,4)", "VWAP偏离%"),
    ("sort_reason", "TEXT", "排序理由"),
    ("sectors", "VARCHAR(200)", "板块"),
    ("risk_factors", "TEXT", "风险因素"),
]


def ensure_new_columns(session):
    """确保 stock_selected_picks 表存在新字段"""
    for col_name, col_type, _ in NEW_PICKS_COLS:
        sql = text(f"""
            ALTER TABLE stock_selected_picks 
            ADD COLUMN IF NOT EXISTS {col_name} {col_type}
        """)
        session.execute(sql)
    session.commit()


def ensure_tick_stats_columns(session):
    """确保 stock_selected_picks 表存在逐笔统计字段"""
    for col_name, col_type, _ in TICK_STATS_COLS:
        sql = text(f"""
            ALTER TABLE stock_selected_picks 
            ADD COLUMN IF NOT EXISTS {col_name} {col_type}
        """)
        session.execute(sql)
    session.commit()


def get_last_n_trading_dates(session, pick_date, n=5):
    """从数据库获取最近N个交易日的日期列表（包含pick_date当天）
    Args:
        session: 数据库会话
        pick_date: 参考日期（通常是最近交易日）
        n: 需要的交易日数量
    Returns:
        list: 最近N个交易日的日期列表（从早到晚排序）
    """
    sql = text("""
        SELECT DISTINCT bar_time::date
        FROM stock_k_data
        WHERE bar_time <= :pick_date AND freq = 'd'
        ORDER BY bar_time DESC
        LIMIT :n
    """)
    rows = session.execute(sql, {"pick_date": pick_date, "n": n}).fetchall()
    dates = [row[0] for row in rows]
    return list(reversed(dates))


def _is_continuous_auction(time_str: str) -> bool:
    """判断是否为连续竞价时段（过滤集合竞价）
    连续竞价：09:30-11:30, 13:00-14:57
    集合竞价：09:15-09:25, 14:57-15:00
    """
    return "09:30" <= time_str <= "11:30" or "13:00" <= time_str <= "14:56"


def calc_tick_stats_for_stock(api, ts_code, trading_dates):
    """计算单只股票最近N天的逐笔成交额统计（仅连续竞价，单位：万元）
    Args:
        api: TdxHq_API 实例
        ts_code: 股票代码（纯数字格式，如'688001'）
        trading_dates: 交易日日期列表
    Returns:
        dict: {'avg_tick_amount': float, 'max_tick_amount': float, 'min_tick_amount': float}
              无数据时返回 None
    """
    date_ints = [int(d.strftime("%Y%m%d")) if hasattr(d, 'strftime') else int(str(d).replace('-', '')) for d in trading_dates]
    market = market_from_code(ts_code)
    
    all_amounts_wan = []
    for date_int in date_ints:
        all_ticks = []
        start = 0
        page_size = 2000
        while True:
            data = api.get_history_transaction_data(market, ts_code, start, page_size, date_int)
            if not data:
                break
            all_ticks.extend(data)
            if len(data) < page_size:
                break
            start += page_size
            if start > 50000:
                break
        
        if all_ticks:
            df = pd.DataFrame(all_ticks).drop_duplicates()
            if not df.empty and 'price' in df.columns and 'vol' in df.columns:
                auction = df[df['time'].apply(_is_continuous_auction)]
                auction = auction[auction['vol'] > 0]
                if not auction.empty:
                    amounts_wan = auction['price'] * auction['vol'] * 100 / 10000.0
                    all_amounts_wan.extend(amounts_wan.tolist())
    
    if not all_amounts_wan:
        return None
    
    amounts_series = pd.Series(all_amounts_wan)
    return {
        "avg_tick_amount": float(amounts_series.mean()),
        "max_tick_amount": float(amounts_series.max()),
        "min_tick_amount": float(amounts_series.min()),
    }


def calc_and_update_tick_stats(session, pick_date, ts_codes):
    """批量计算逐笔统计并更新数据库
    Args:
        session: 数据库会话
        pick_date: 选股日期
        ts_codes: 股票代码列表
    """
    trading_dates = get_last_n_trading_dates(session, pick_date, n=5)
    if not trading_dates:
        print("  无法获取最近5个交易日，跳过逐笔统计")
        return

    date_strs = [str(d) for d in trading_dates]
    print(f"  逐笔统计交易日: {', '.join(date_strs)}")

    api = connect_pytdx()
    try:
        success_count = 0
        fail_count = 0
        error_count = 0

        for ts_code in tqdm(ts_codes, desc="逐笔统计", position=0):
            try:
                stats = calc_tick_stats_for_stock(api, ts_code, trading_dates)
                if stats:
                    session.execute(text("""
                        UPDATE stock_selected_picks
                        SET avg_tick_amount = :avg,
                            max_tick_amount = :max,
                            min_tick_amount = :min
                        WHERE pick_date = :pick_date AND ts_code = :ts_code
                    """), {
                        "pick_date": pick_date,
                        "ts_code": ts_code,
                        "avg": stats["avg_tick_amount"],
                        "max": stats["max_tick_amount"],
                        "min": stats["min_tick_amount"],
                    })
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                error_count += 1
                print(f"\n  [{ts_code}] 逐笔统计异常: {e}")
                tqdm.write(f"  [{ts_code}] {traceback.format_exc()}")

        session.commit()
        print(f"\n  逐笔统计完成: {success_count} 只有数据, {fail_count} 只无数据, {error_count} 只异常")
    finally:
        api.disconnect()


def find_csv_file():
    """查找 /root/trading/selection/ 目录下唯一的CSV文件"""
    csv_files = glob.glob(os.path.join(SELECTION_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{SELECTION_DIR} 目录下未找到CSV文件")
    if len(csv_files) > 1:
        raise ValueError(f"{SELECTION_DIR} 目录下存在多个CSV文件: {csv_files}")
    return csv_files[0]


def get_latest_trading_date(session):
    """从数据库获取最近一个交易日"""
    sql = text("""
        SELECT MAX(selection_date) FROM stock_selection_results
        WHERE selection_date IS NOT NULL
    """)
    result = session.execute(sql).scalar()
    if result:
        return result

    sql = text("""
        SELECT MAX(trade_date)::date FROM stock_k_data
        WHERE trade_date IS NOT NULL
    """)
    result = session.execute(sql).scalar()
    if not result:
        raise ValueError("无法从数据库获取最近交易日")
    return result


def load_signal_dates(session, ts_codes, pick_date):
    """从 stock_selection_results 表查询信号日期
    Args:
        session: 数据库会话
        ts_codes: 股票代码列表（带后缀格式，如'000938.SZ'或'600000.SH'）
        pick_date: 选股日期
    Returns:
        dict: {ts_code: signal_date} 映射，无信号的股票不在此字典中
    """
    if not ts_codes:
        return {}
    
    # 过滤空值
    ts_codes = [c for c in ts_codes if c and c.strip()]
    
    placeholders = ",".join([f"'{c}'" for c in ts_codes])
    sql = text(f"""
        SELECT ts_code, signal_date
        FROM stock_selection_results
        WHERE selection_date = '{pick_date}'
        AND ts_code IN ({placeholders})
        AND signal_date IS NOT NULL
    """)
    result = session.execute(sql).fetchall()
    
    return {row[0]: row[1] for row in result}


def load_csv_to_db(csv_path, pick_date, session):
    """读取CSV并导入数据库"""
    ensure_tick_stats_columns(session)
    ensure_new_columns(session)

    df = pd.read_csv(csv_path, dtype={"代码": str})

    expected_cols = ["代码", "名称", "投资推荐分", "推荐档位", "操作建议", "主题标签",
                     "估值判断", "定位", "持续性", "业绩趋势", "上涨空间%",
                     "周线Z分", "周线VolZ", "日线VolZ", "VWAP偏离%", "排序理由", "板块", "炒作逻辑", "风险因素"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少字段: {missing}")

    ts_codes = [str(row["代码"]).strip() for _, row in df.iterrows()]
    signal_map = load_signal_dates(session, ts_codes, pick_date)

    # 幂等性：删除同日期旧数据
    session.execute(text("DELETE FROM stock_selected_picks WHERE pick_date = :d"), {"d": pick_date})
    session.commit()

    # 批量插入
    records = []
    for rank_idx, (_, row) in enumerate(df.iterrows(), start=1):
        ts_code = str(row["代码"]).strip()
        score_raw = row["投资推荐分"]
        upside_raw = row["上涨空间%"]
        weekly_z_raw = row["周线Z分"]
        weekly_vol_z_raw = row["周线VolZ"]
        daily_vol_z_raw = row["日线VolZ"]
        vwap_dev_raw = row["VWAP偏离%"]

        records.append({
            "pick_date": pick_date,
            "rank": rank_idx,
            "ts_code": ts_code,
            "stock_name": str(row["名称"]).strip(),
            "recommend_score": float(score_raw) if pd.notna(score_raw) else None,
            "recommend_grade": str(row["推荐档位"]).strip() if pd.notna(row["推荐档位"]) else None,
            "action_advice": str(row["操作建议"]).strip() if pd.notna(row["操作建议"]) else None,
            "theme_tags": str(row["主题标签"]).strip() if pd.notna(row["主题标签"]) else None,
            "valuation_judge": str(row["估值判断"]).strip() if pd.notna(row["估值判断"]) else None,
            "positioning": str(row["定位"]).strip() if pd.notna(row["定位"]) else None,
            "sustainability": str(row["持续性"]).strip() if pd.notna(row["持续性"]) else None,
            "profit_trend": str(row["业绩趋势"]).strip() if pd.notna(row["业绩趋势"]) else None,
            "upside_pct": float(upside_raw) if pd.notna(upside_raw) else None,
            "weekly_z": float(weekly_z_raw) if pd.notna(weekly_z_raw) else None,
            "weekly_vol_z": float(weekly_vol_z_raw) if pd.notna(weekly_vol_z_raw) else None,
            "daily_vol_z": float(daily_vol_z_raw) if pd.notna(daily_vol_z_raw) else None,
            "vwap_deviation": float(vwap_dev_raw) if pd.notna(vwap_dev_raw) else None,
            "sort_reason": str(row["排序理由"]).strip() if pd.notna(row["排序理由"]) else None,
            "sectors": str(row["板块"]).strip() if pd.notna(row["板块"]) else None,
            "logic": str(row["炒作逻辑"]).strip() if pd.notna(row["炒作逻辑"]) else None,
            "risk_factors": str(row["风险因素"]).strip() if pd.notna(row["风险因素"]) else None,
            "signal_date": signal_map.get(ts_code),
            "score": None,
            "theme": None,
            "position": None,
            "report_period": None,
            "detail": None,
        })

    session.execute(text("""
        INSERT INTO stock_selected_picks
        (pick_date, rank, ts_code, stock_name,
         recommend_score, recommend_grade, action_advice, theme_tags,
         valuation_judge, positioning, sustainability, profit_trend,
         upside_pct, weekly_z, weekly_vol_z, daily_vol_z, vwap_deviation,
         sort_reason, sectors, logic, risk_factors, signal_date,
         score, theme, position, report_period, detail)
        VALUES (:pick_date, :rank, :ts_code, :stock_name,
                :recommend_score, :recommend_grade, :action_advice, :theme_tags,
                :valuation_judge, :positioning, :sustainability, :profit_trend,
                :upside_pct, :weekly_z, :weekly_vol_z, :daily_vol_z, :vwap_deviation,
                :sort_reason, :sectors, :logic, :risk_factors, :signal_date,
                :score, :theme, :position, :report_period, :detail)
    """), records)
    session.commit()

    has_signal = sum(1 for r in records if r["signal_date"] is not None)
    print(f"  其中 {has_signal} 只股票有信号日期，{len(records) - has_signal} 只无信号（signal_date=NULL）")

    return ts_codes


def main():
    csv_path = find_csv_file()
    print(f"找到CSV文件: {csv_path}")

    with get_session() as session:
        pick_date = get_latest_trading_date(session)
        print(f"最近交易日: {pick_date}")

        ts_codes = load_csv_to_db(csv_path, pick_date, session)
        print(f"成功导入 {len(ts_codes)} 条数据到 stock_selected_picks 表，日期: {pick_date}")

        print("\n开始计算逐笔成交额统计...")
        calc_and_update_tick_stats(session, pick_date, ts_codes)


if __name__ == "__main__":
    main()

