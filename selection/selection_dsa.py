#!/usr/bin/env python3
"""
DSA + BBMacd V型反转选股脚本（基于周线数据）

Purpose: 检测周线BBMacd V型反转触发点，记录24因子状态和未来标签
Inputs: stock_k_data (周线K线数据)
Outputs: stock_dsa_vreversal_results (选股结果)
How to Run:
    python selection/selection_dsa.py --ts-code 600547              # 单股模式
    python selection/selection_dsa.py --date 2026-04-10             # 全股日扫
    python selection/selection_dsa.py --backfill --start-date 2024-01-01 --end-date 2026-04-30  # 回补
Side Effects: 写入 stock_dsa_vreversal_results 表

================================================================================
【选股逻辑】

触发条件：BBMacd V型反转
  bbmacd[t] > bbmacd[t-1] 且 bbmacd[t-1] < bbmacd[t-2]
  （bbmacd在t-1处形成局部最低点后拐头向上）

状态记录：触发时记录24个因子的值

未来标签（仅用于训练，不可用于选股决策）：
  1. 反转高点：从触发点t向后，找下一次dsa_dir从1变为-1的位置t_rev，
     取t到t_rev区间内high的最大值作为"反转高点股价"，计算相对触发时close的收益率，
     记录反转高点所在bar距触发点的bar数
  2. 中间最低点：取t到t_rev区间内low的最小值作为"中间最低点股价"，
     计算相对触发时close的收益率，记录最低点所在bar距触发点的bar数
  3. 若到数据末尾仍未出现dsa_dir 1→-1反转，标签记为NaN

【三个入口】

入口1 --ts-code：指定个股，输出该股所有历史触发点
入口2 --date：指定日期全股扫描
入口3 --backfill：按周五列表逐日回补

【数据源】
仅使用数据库 stock_k_data 表的周线(freq='w')数据
================================================================================
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from features.dsa_bbmacd_24factors_viewer import (
    compute_dsa,
    compute_bbmacd,
    compute_24_factors,
    DSAConfig,
)

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "stock_dsa_vreversal_results"
WEEKLY_BARS = 600
DSA_CFG = DSAConfig(prd=50, base_apt=20.0, use_adapt=False, vol_bias=10.0, atr_len=50)

FACTOR_COLUMNS = [
    "dsa_dir",
    "prev_pivot_code",
    "last_confirmed_high",
    "last_confirmed_low",
    "dsa_pivot_pos_01",
    "ret_to_last_high_pct",
    "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct",
    "current_stage_bars",
    "prev_stage_bars",
    "bars_since_last_high",
    "bars_since_last_low",
    "prev_stage_amp_pct",
    "current_stage_ret_pct",
    "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct",
    "bbmacd",
    "bbmacd_minus_avg",
    "bbmacd_state",
    "bbmacd_band_pos_01",
    "bbmacd_bandwidth_zscore",
    "bbmacd_cross_upper",
    "bbmacd_cross_lower",
    "trend_align_momo",
]


def normalize_ts_code(ts_code: str) -> str:
    return str(ts_code).strip().upper().split(".")[0]


def get_kline_data_db(
    ts_code: str, freq: str = "w", bars: int = WEEKLY_BARS, end_date: Optional[date] = None
) -> pd.DataFrame:
    """从 stock_k_data 表获取周线K线数据"""
    symbol = normalize_ts_code(ts_code)
    if end_date is not None:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz)
              AND freq = :freq AND DATE(bar_time) <= :end_date
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            "ts_code": symbol,
            "ts_code_sh": f"{symbol}.SH",
            "ts_code_sz": f"{symbol}.SZ",
            "freq": freq,
            "bars": bars,
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
    else:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz)
              AND freq = :freq
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            "ts_code": symbol,
            "ts_code_sh": f"{symbol}.SH",
            "ts_code_sz": f"{symbol}.SZ",
            "freq": freq,
            "bars": bars,
        }

    df = pd.read_sql(text(sql), engine, params=params)
    if not df.empty:
        df = df.sort_values("bar_time").set_index("bar_time")
    return df


def get_stock_list_db(target_date: date) -> List[str]:
    """获取指定日期有周线数据的股票代码列表"""
    sql = text(
        """
        SELECT DISTINCT ts_code
        FROM stock_k_data
        WHERE freq = 'w' AND DATE(bar_time) = :target_date
        """
    )
    with engine.connect() as conn:
        result = conn.execute(sql, {"target_date": target_date.strftime("%Y-%m-%d")})
        codes = [normalize_ts_code(row[0]) for row in result]
    return codes


def get_all_weekly_stock_list() -> List[str]:
    """获取所有有周线数据的股票代码列表"""
    sql = text(
        """
        SELECT DISTINCT ts_code
        FROM stock_k_data
        WHERE freq = 'w'
        """
    )
    with engine.connect() as conn:
        result = conn.execute(sql)
        codes = list(set(normalize_ts_code(row[0]) for row in result))
    return codes


def bar_time_to_date(bar_time) -> date:
    """将 bar_time 转为 date，直接用触发bar的日期作为 selection_date"""
    if hasattr(bar_time, "date"):
        return bar_time.date()
    if isinstance(bar_time, date):
        return bar_time
    return pd.Timestamp(bar_time).date()


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """批量获取股票名称"""
    if not ts_codes:
        return {}
    placeholders = ", ".join([f"'{c}'" for c in ts_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return {row[0]: row[1] for row in result}


def detect_vreversal_triggers(bbmacd: pd.Series) -> np.ndarray:
    """返回布尔数组，标记V型反转触发点: bbmacd[t]>bbmacd[t-1] 且 bbmacd[t-1]<bbmacd[t-2]"""
    t_gt_t1 = bbmacd > bbmacd.shift(1)
    t1_lt_t2 = bbmacd.shift(1) < bbmacd.shift(2)
    valid = bbmacd.notna() & bbmacd.shift(1).notna() & bbmacd.shift(2).notna()
    return (t_gt_t1 & t1_lt_t2 & valid).to_numpy()


def compute_future_labels(
    index: pd.DatetimeIndex,
    dsa_dir: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    trigger_idx: int,
    close_at_trigger: float,
) -> Dict:
    """从trigger_idx向后扫描，计算未来标签

    若找到dsa_dir 1→-1反转，标签区间为 [trigger_idx, t_rev]；
    若到数据末尾仍未反转，标签区间为 [trigger_idx, 末尾]，用区间内最高/最低价填充。
    """
    n = len(dsa_dir)
    end_idx = n - 1
    found_reversal = False
    for i in range(trigger_idx + 1, n):
        if dsa_dir[i - 1] == 1 and dsa_dir[i] == -1:
            end_idx = i
            found_reversal = True
            break

    seg_high = high[trigger_idx : end_idx + 1]
    seg_low = low[trigger_idx : end_idx + 1]
    rev_high = float(np.nanmax(seg_high))
    int_low = float(np.nanmin(seg_low))
    rev_high_rel = int(np.nanargmax(seg_high))
    int_low_rel = int(np.nanargmin(seg_low))
    rev_high_idx = trigger_idx + rev_high_rel
    int_low_idx = trigger_idx + int_low_rel

    result = {
        "next_reversal_high_price": rev_high,
        "next_reversal_high_ret": (rev_high - close_at_trigger) / close_at_trigger
        if close_at_trigger != 0
        else None,
        "next_reversal_bar_time": index[end_idx] if found_reversal else None,
        "bars_to_reversal_high": rev_high_rel,
        "interim_low_price": int_low,
        "interim_low_ret": (int_low - close_at_trigger) / close_at_trigger
        if close_at_trigger != 0
        else None,
        "interim_low_bar_time": index[int_low_idx],
        "bars_to_interim_low": int_low_rel,
        "bars_to_reversal": end_idx - trigger_idx if found_reversal else None,
    }
    return result


def process_stock(ts_code: str, end_date: Optional[date] = None) -> List[Dict]:
    """处理单只股票，返回触发点记录列表

    始终用完整历史数据计算24因子和未来标签，end_date仅用于筛选触发点范围。
    """
    df = get_kline_data_db(ts_code, freq="w", bars=WEEKLY_BARS, end_date=None)
    if df.empty or len(df) < 50:
        return []

    try:
        dsa_df, _, _ = compute_dsa(df, DSA_CFG)
    except Exception:
        return []

    if dsa_df.empty:
        return []

    try:
        bb_df = compute_bbmacd(df)
    except Exception:
        return []

    merged = pd.concat([df, dsa_df, bb_df], axis=1)
    try:
        factors_df = compute_24_factors(merged)
    except Exception:
        return []

    triggers = detect_vreversal_triggers(merged["bbmacd"])
    trigger_indices = np.where(triggers)[0]
    if len(trigger_indices) == 0:
        return []

    dsa_dir_arr = factors_df["dsa_dir"].to_numpy(float)
    high_arr = merged["high"].to_numpy(float)
    low_arr = merged["low"].to_numpy(float)

    records = []
    for idx in trigger_indices:
        trigger_bt = merged.index[idx]
        trigger_date = trigger_bt.date() if hasattr(trigger_bt, "date") else trigger_bt
        if end_date is not None and trigger_date > end_date:
            continue
        close_at_t = float(merged["close"].iloc[idx])
        labels = compute_future_labels(
            merged.index, dsa_dir_arr, high_arr, low_arr, idx, close_at_t
        )
        row = factors_df.iloc[idx]
        record = {
            "ts_code": ts_code,
            "trigger_bar_time": trigger_bt,
            "trigger_close": close_at_t,
        }
        for col in FACTOR_COLUMNS:
            val = row.get(col)
            if isinstance(val, (pd.Series, np.ndarray)):
                val = val.iloc[0] if len(val) else None
            record[col] = float(val) if pd.notna(val) else None
        record.update(labels)
        records.append(record)
    return records


def ensure_table_exists():
    """确保选股结果表存在"""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {SELECTION_TABLE} (
        id              BIGSERIAL PRIMARY KEY,
        selection_date  DATE NOT NULL,
        ts_code         VARCHAR(20) NOT NULL,
        stock_name      VARCHAR(50),
        freq            VARCHAR(5) DEFAULT 'w',
        trigger_bar_time TIMESTAMP NOT NULL,
        trigger_close   FLOAT,
        dsa_dir                                 FLOAT,
        prev_pivot_code                         FLOAT,
        last_confirmed_high                     FLOAT,
        last_confirmed_low                      FLOAT,
        dsa_pivot_pos_01                        FLOAT,
        ret_to_last_high_pct                    FLOAT,
        ret_to_last_low_pct                     FLOAT,
        price_vs_dsa_vwap_pct                   FLOAT,
        current_stage_bars                      FLOAT,
        prev_stage_bars                         FLOAT,
        bars_since_last_high                    FLOAT,
        bars_since_last_low                     FLOAT,
        prev_stage_amp_pct                      FLOAT,
        current_stage_ret_pct                   FLOAT,
        current_stage_amp_pct                   FLOAT,
        current_pullback_from_stage_extreme_pct FLOAT,
        bbmacd                                  FLOAT,
        bbmacd_minus_avg                        FLOAT,
        bbmacd_state                            INT,
        bbmacd_band_pos_01                      FLOAT,
        bbmacd_bandwidth_zscore                 FLOAT,
        bbmacd_cross_upper                      FLOAT,
        bbmacd_cross_lower                      FLOAT,
        trend_align_momo                        FLOAT,
        next_reversal_high_price   FLOAT,
        next_reversal_high_ret     FLOAT,
        next_reversal_bar_time     TIMESTAMP,
        bars_to_reversal_high      INT,
        interim_low_price          FLOAT,
        interim_low_ret            FLOAT,
        interim_low_bar_time       TIMESTAMP,
        bars_to_interim_low        INT,
        bars_to_reversal           INT,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code, trigger_bar_time)
    );
    CREATE INDEX IF NOT EXISTS idx_dsa_vrev_sel_date   ON {SELECTION_TABLE}(selection_date);
    CREATE INDEX IF NOT EXISTS idx_dsa_vrev_ts_code    ON {SELECTION_TABLE}(ts_code);
    CREATE INDEX IF NOT EXISTS idx_dsa_vrev_trigger    ON {SELECTION_TABLE}(trigger_bar_time);
    CREATE INDEX IF NOT EXISTS idx_dsa_vrev_rev_ret    ON {SELECTION_TABLE}(next_reversal_high_ret);
    CREATE INDEX IF NOT EXISTS idx_dsa_vrev_low_ret    ON {SELECTION_TABLE}(interim_low_ret);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def save_to_database(df: pd.DataFrame, selection_date: date) -> int:
    """保存选股结果到数据库（幂等：先删后插）"""
    if df.empty:
        return 0

    ensure_table_exists()

    with engine.connect() as conn:
        delete_sql = text(
            f"DELETE FROM {SELECTION_TABLE} WHERE selection_date = :selection_date"
        )
        result = conn.execute(delete_sql, {"selection_date": selection_date})
        conn.commit()
        if result.rowcount > 0:
            print(f"    清除旧数据: {result.rowcount} 条")

    insert_df = df.copy()
    insert_df["selection_date"] = selection_date
    insert_df.to_sql(SELECTION_TABLE, engine, if_exists="append", index=False)
    return len(insert_df)


def run_single_stock(
    ts_code: str, selection_date: Optional[date] = None, save_to_db: bool = True
) -> pd.DataFrame:
    """入口1：处理指定个股，输出该股所有历史触发点"""
    ts_code = normalize_ts_code(ts_code)
    sel_date = selection_date or date.today()

    print(f"\n{'=' * 80}")
    print(f"单股模式: {ts_code}")
    print(f"截止日期: {sel_date}")
    print(f"{'=' * 80}")

    records = process_stock(ts_code, end_date=selection_date)
    if not records:
        print("  未找到触发点")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["selection_date"] = sel_date
    df["ts_code"] = ts_code

    stock_names = batch_get_stock_names([ts_code])
    df["stock_name"] = stock_names.get(ts_code, "")

    print(f"  触发点数: {len(df)}")
    if not df.empty:
        has_label = df["next_reversal_high_ret"].notna().sum()
        print(f"  有未来标签: {has_label}")
        print(f"\n  最近5个触发点:")
        display_cols = [
            "trigger_bar_time",
            "trigger_close",
            "bbmacd",
            "dsa_dir",
            "next_reversal_high_ret",
            "interim_low_ret",
        ]
        avail_cols = [c for c in display_cols if c in df.columns]
        print(df[avail_cols].tail().to_string(index=False))

    if save_to_db:
        saved = save_to_database(df, sel_date)
        print(f"\n  保存到数据库: {saved} 条")

    return df


def run_full_scan(selection_date: date, save_to_db: bool = True) -> pd.DataFrame:
    """入口2：扫描该日期有周线数据的全部股票"""
    print(f"\n{'=' * 80}")
    print(f"全股扫描模式")
    print(f"选股日期: {selection_date}")
    print(f"{'=' * 80}")

    stock_list = get_stock_list_db(selection_date)
    print(f"  找到 {len(stock_list)} 只有周线数据的股票")

    if not stock_list:
        return pd.DataFrame()

    all_records: List[Dict] = []
    for ts_code in tqdm(stock_list, desc="全股扫描", unit="只"):
        try:
            records = process_stock(ts_code, end_date=selection_date)
            for r in records:
                r["selection_date"] = selection_date
                r["ts_code"] = ts_code
            all_records.extend(records)
        except Exception:
            continue

    if not all_records:
        print("  未找到任何触发点")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    stock_names = batch_get_stock_names(df["ts_code"].unique().tolist())
    df["stock_name"] = df["ts_code"].map(stock_names)

    print(f"\n  总触发点数: {len(df)}")
    print(f"  涉及股票数: {df['ts_code'].nunique()}")
    has_label = df["next_reversal_high_ret"].notna().sum()
    print(f"  有未来标签: {has_label}")

    if save_to_db:
        saved = save_to_database(df, selection_date)
        print(f"\n  保存到数据库: {saved} 条")

    return df


def run_backfill(
    start_date: date, end_date: date, save_to_db: bool = True, max_stocks: int = 0
) -> Dict[date, int]:
    """入口3：遍历每只股票只算一次，按 trigger_bar_time 映射到对应周五的 selection_date

    优化逻辑：
      旧：for friday in fridays: for stock in stocks: process(stock, end_date=friday)
          同一只股票在不同周五被重复计算 DSA/BBMacd/24因子，效率极低
      新：for stock in stocks: process(stock) 一次，将触发点按 trigger_bar_time 映射到周五
          每只股票只拉一次数据、只算一次，然后按 selection_date 分组保存
    """
    print(f"\n{'=' * 80}")
    print(f"回补模式（遍历个股）")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"{'=' * 80}")

    stock_list = get_all_weekly_stock_list()
    if max_stocks > 0:
        stock_list = stock_list[:max_stocks]
    print(f"  共 {len(stock_list)} 只股票需要处理\n")

    all_records: List[Dict] = []
    for ts_code in tqdm(stock_list, desc="回补进度", unit="只"):
        try:
            records = process_stock(ts_code, end_date=None)
        except Exception:
            continue
        for r in records:
            r["ts_code"] = ts_code
            trigger_bt = r.get("trigger_bar_time")
            if trigger_bt is None:
                continue
            sel_date = bar_time_to_date(trigger_bt)
            if start_date <= sel_date <= end_date:
                r["selection_date"] = sel_date
                all_records.append(r)

    if not all_records:
        print("  未找到任何触发点")
        return {}

    df = pd.DataFrame(all_records)

    stock_names = batch_get_stock_names(df["ts_code"].unique().tolist())
    df["stock_name"] = df["ts_code"].map(stock_names)

    results: Dict[date, int] = {}
    if save_to_db:
        for sel_date, group_df in df.groupby("selection_date"):
            saved = save_to_database(group_df, sel_date)
            results[sel_date] = saved
    else:
        for sel_date, group_df in df.groupby("selection_date"):
            results[sel_date] = len(group_df)

    print(f"\n{'=' * 80}")
    print("回补完成")
    print(f"  总股票数: {len(stock_list)}")
    print(f"  总触发点数: {len(df)}")
    print(f"  覆盖周五数: {len(results)}")
    print(f"  有结果周五数: {sum(1 for v in results.values() if v > 0)}")
    print(f"{'=' * 80}")

    return results


def parse_date(date_str: str) -> date:
    """解析日期字符串"""
    for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DSA + BBMacd V型反转选股工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_dsa.py --ts-code 600547                     # 单股模式
  python selection/selection_dsa.py --ts-code 600547 --date 2026-04-10   # 单股+截止日期
  python selection/selection_dsa.py --date 2026-04-10                    # 全股日扫
  python selection/selection_dsa.py                                      # 默认今天全股扫描
  python selection/selection_dsa.py --backfill --start-date 2024-01-01   # 回补
        """,
    )
    parser.add_argument("--ts-code", help="入口1：指定个股代码，如 600547")
    parser.add_argument("--date", help="选股日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument(
        "--backfill", action="store_true", help="入口3：全量回补模式"
    )
    parser.add_argument(
        "--start-date", help="回补开始日期 (YYYY-MM-DD)，与 --backfill 一起使用"
    )
    parser.add_argument(
        "--end-date", help="回补结束日期 (YYYY-MM-DD)，与 --backfill 一起使用，默认今天"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="不保存到数据库（仅输出到终端）"
    )
    parser.add_argument("--csv-out", default="", help="额外输出CSV文件")
    parser.add_argument(
        "--max-stocks", type=int, default=0, help="回补时最多处理的股票数（0=全部，用于测试）"
    )

    args = parser.parse_args()
    save_to_db = not args.no_save

    if args.backfill:
        if not args.start_date:
            print("错误: 使用 --backfill 时必须提供 --start-date")
            sys.exit(1)
        try:
            start_date = parse_date(args.start_date)
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
        end_date = parse_date(args.end_date) if args.end_date else date.today()
        run_backfill(start_date, end_date, save_to_db=save_to_db, max_stocks=args.max_stocks)
        return

    if args.date:
        try:
            selection_date = parse_date(args.date)
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
    else:
        selection_date = date.today()

    if args.ts_code:
        df = run_single_stock(args.ts_code, selection_date, save_to_db=save_to_db)
    else:
        df = run_full_scan(selection_date, save_to_db=save_to_db)

    if args.csv_out and not df.empty:
        df.to_csv(args.csv_out, encoding="utf-8-sig", index=False)
        print(f"CSV 已生成: {args.csv_out}")


if __name__ == "__main__":
    main()
