# cache_generator.py
"""
股票概念缓存生成器

功能：
1. 更新股票概念缓存：通过 pywencai 获取所有 A 股的概念、人气排名、市值等数据，保存到 Excel 文件
2. 获取指定日期的人气排名：查询指定交易日的人气排名数据并打印
3. 扫描 1 年内人气排名：遍历 1 年内每个交易日的人气排名并打印前 10 名

Usage:
    # 更新股票概念缓存
    python cores/cache_generator.py --update-cache
    
    # 获取指定日期的人气排名
    python cores/cache_generator.py --date 2024-03-01
    python cores/cache_generator.py --date 2024-12-31
    
    # 扫描 1 年内的人气排名（从今天往前推 365 天）
    python cores/cache_generator.py --scan-popularity
    
    # 扫描到指定日期为止的 1 年
    python cores/cache_generator.py --scan-popularity --end-date 2024-03-01
    
    # 创建数据库表
    python cores/cache_generator.py --create-table
    
    # 扫描并保存到数据库（增量更新）
    python cores/cache_generator.py --scan-popularity --save-to-db
    
    # 扫描并保存到数据库（全量覆盖）
    python cores/cache_generator.py --scan-popularity --save-to-db --no-incremental
    
    # 获取指定日期数据并保存到数据库
    python cores/cache_generator.py --date 2024-03-01 --save-to-db
    
    # 查看帮助
    python cores/cache_generator.py --help

Examples:
    # 示例 1: 获取 2024 年 3 月 1 日的人气排名前 20
    python cores/cache_generator.py --date 2024-03-01
    
    # 示例 2: 扫描 2024 年全年的交易日人气排名
    python cores/cache_generator.py --scan-popularity --end-date 2024-12-31
    
    # 示例 3: 更新最新的股票概念缓存
    python cores/cache_generator.py --update-cache

Side Effects:
    - --update-cache: 会在项目根目录生成/更新 stock_concepts_cache.xlsx 文件
    - --date: 仅打印输出，不修改任何文件
    - --date --save-to-db: 将指定日期数据写入数据库 stock_popularity_rank 表
    - --scan-popularity: 仅打印输出，不修改任何文件
    - --scan-popularity --save-to-db: 将所有日期数据写入数据库（默认增量更新）
    - --scan-popularity --save-to-db --no-incremental: 全量覆盖写入数据库
    - --create-table: 创建数据库表 stock_popularity_rank

Dependencies:
    - pywencai: 用于获取问财数据
    - qstock: 用于获取交易日历（可选，失败时会自动降级到周末判断法）
"""
import pywencai as wc
import pandas as pd
import re
import sys
import os
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cache_generator")

def _transform_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    """[内部函数] 负责将原始DataFrame转换为干净、标准化的格式以供入库。"""
    concept_col = next((col for col in raw_df.columns if '所属概念' in col), None)
    rank_col = next((col for col in raw_df.columns if re.search(r'个股热度排名', col)), None)
    market_cap_col = next((col for col in raw_df.columns if re.search(r'流通|限售', col)), None)
    total_market_cap_col = next((col for col in raw_df.columns if '总市值' in col), None)
    industry_col = next((col for col in raw_df.columns if '所属同花顺行业' in col or '所属同花顺二级行业' in col), None)
    industry_pe_col = next((col for col in raw_df.columns if '行业市盈率' in col), None)

    required_cols = [concept_col, rank_col, market_cap_col]
    if not all(required_cols):
        logger.error("在数据源中未能找到所有必需的列（概念、人气、市值）")
        return None
    
    logger.info(f"成功匹配到列: 概念='{concept_col}', 人气='{rank_col}', 流通市值='{market_cap_col}'")
    logger.info(f"新增列: 总市值='{total_market_cap_col}', 行业分类='{industry_col}', 行业平均PE='{industry_pe_col}'")
    
    base_cols = ['股票代码', '股票简称', concept_col, rank_col, market_cap_col]
    rename_map = {
        '股票简称': 'name', 
        concept_col: 'concepts',
        rank_col: 'popularity_rank',
        market_cap_col: 'market_cap'
    }
    
    if total_market_cap_col:
        base_cols.append(total_market_cap_col)
        rename_map[total_market_cap_col] = 'total_market_cap'
    if industry_col:
        base_cols.append(industry_col)
        rename_map[industry_col] = 'industry'
    if industry_pe_col:
        base_cols.append(industry_pe_col)
        rename_map[industry_pe_col] = 'industry_pe'
        
    cache_df = raw_df[base_cols].copy()
    cache_df.rename(columns=rename_map, inplace=True)

    def format_ts_code(code):
        code_str = str(code)
        if '.' in code_str: return code_str.upper()
        code_padded = code_str.zfill(6)
        return f"{code_padded}.SZ" if code_padded.startswith(('0', '3')) else f"{code_padded}.SH"
    cache_df['ts_code'] = cache_df['股票代码'].apply(format_ts_code)
    
    original_count = len(cache_df)
    cache_df['popularity_rank'] = pd.to_numeric(cache_df['popularity_rank'], errors='coerce')
    cache_df['market_cap'] = pd.to_numeric(cache_df['market_cap'], errors='coerce')
    if 'total_market_cap' in cache_df.columns:
        cache_df['total_market_cap'] = pd.to_numeric(cache_df['total_market_cap'], errors='coerce')
    if 'industry_pe' in cache_df.columns:
        cache_df['industry_pe'] = pd.to_numeric(cache_df['industry_pe'], errors='coerce')
    cache_df.dropna(subset=['popularity_rank', 'market_cap'], inplace=True)
    cleaned_count = len(cache_df)
    
    if original_count > cleaned_count:
        logger.info(f"数据清洗：发现并删除了 {original_count - cleaned_count} 行无效数据")
    
    if not cache_df.empty:
        cache_df['popularity_rank'] = cache_df['popularity_rank'].astype(int)
    
    output_cols = ['ts_code', 'name', 'concepts', 'popularity_rank', 'market_cap']
    if 'total_market_cap' in cache_df.columns:
        output_cols.append('total_market_cap')
    if 'industry' in cache_df.columns:
        output_cols.append('industry')
    if 'industry_pe' in cache_df.columns:
        output_cols.append('industry_pe')
        
    return cache_df[output_cols]

def update_cache_from_pywencai():
    """
    使用pywencai自动获取所有A股的概念、人气和市值，并将其保存到Excel文件。
    """
    logger.info("正在通过问财获取所有A股的概念、人气及市值数据...")
    try:
        query_text = '非st，主板或创业板或科创板，所属概念，人气排名，流通市值，总市值，行业分类，行业平均PE'
        res = wc.get(query=query_text, loop=True, sleep=2, cookie=None)
        
        if res is None or res.empty:
            logger.error("未能从问财获取到数据。可能是Cookie已过期。")
            return

        logger.info(f"成功获取到 {len(res)} 条原始数据，正在处理...")
        transformed_df = _transform_raw_data(res)
    
        if transformed_df is not None and not transformed_df.empty:
            logger.info("正在将缓存数据写入Excel文件")
            
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            excel_path = os.path.join(project_root, 'stock_concepts_cache.xlsx')
            
            transformed_df.to_excel(excel_path, index=False, engine='openpyxl')
            logger.info(f"成功保存数据到: {excel_path}")
            logger.info(f"共保存 {len(transformed_df)} 条股票数据")
        else:
            logger.warning("数据转换失败或结果为空，未写入Excel文件。")

    except Exception as e:
        logger.error(f"生成概念缓存时发生错误: {e}")
        raise

def get_popularity_rank_by_date(target_date: str):
    """
    获取指定日期的股票人气排名数据
    
    Args:
        target_date: 目标日期，格式 'YYYY-MM-DD'
    
    Returns:
        DataFrame: 包含人气排名数据的 DataFrame
    """
    logger.info(f"正在获取 {target_date} 的人气排名数据...")
    try:
        query_text = f'{target_date} 科创板或创业板或主板非 st 人气排名'
        logger.info(f"问财问句：{query_text}")
        res = wc.get(query=query_text, loop=True, sleep=2, cookie=None)
        
        if res is None or res.empty:
            logger.warning(f"未能从问财获取到 {target_date} 的数据")
            return None
        
        logger.info(f"成功获取到 {len(res)} 条数据")
        return res
        
    except Exception as e:
        logger.error(f"获取 {target_date} 人气排名时发生错误：{e}")
        return None


def _transform_rank_data(raw_df: pd.DataFrame, trade_date: str) -> pd.DataFrame | None:
    """
    转换人气排名数据为入库格式（只保留必要字段）
    
    Args:
        raw_df: 原始数据 DataFrame
        trade_date: 交易日期
        
    Returns:
        转换后的 DataFrame（只包含 trade_date, ts_code, name, rank）
    """
    if raw_df is None or raw_df.empty:
        return None
    
    # 匹配列名（支持多种格式）
    rank_col = next((col for col in raw_df.columns if '排名' in col or '热度排名' in col), None)
    name_col = next((col for col in raw_df.columns if '股票简称' in col or '股票名称' in col), None)
    
    if not all([rank_col, name_col]):
        logger.error(f"未能找到必需的列（排名、名称）")
        logger.error(f"可用列：{raw_df.columns.tolist()}")
        return None
    
    def format_ts_code(code):
        code_str = str(code)
        if '.' in code_str:
            return code_str.upper()
        code_padded = code_str.zfill(6)
        if code_padded.startswith(('0', '3')):
            return f"{code_padded}.SZ"
        else:
            return f"{code_padded}.SH"
    
    result_df = pd.DataFrame({
        'trade_date': trade_date,
        'ts_code': raw_df['股票代码'].apply(format_ts_code),
        'name': raw_df[name_col],
        'rank': pd.to_numeric(raw_df[rank_col], errors='coerce').astype(int)
    })
    
    result_df.dropna(subset=['rank'], inplace=True)
    
    return result_df[['trade_date', 'ts_code', 'name', 'rank']]


def save_popularity_to_db(df: pd.DataFrame, session) -> int:
    """
    将人气排名数据保存到数据库
    
    Args:
        df: 数据 DataFrame
        session: 数据库会话
        
    Returns:
        写入的行数
    """
    from app.db import bulk_upsert
    from sqlalchemy import Table, MetaData
    
    if df is None or df.empty:
        return 0
    
    metadata = MetaData()
    table = Table('stock_popularity_rank', metadata, autoload_with=session.bind)
    
    unique_keys = ['trade_date', 'ts_code']
    count = bulk_upsert(session, type('Model', (), {'__tablename__': 'stock_popularity_rank'}), df, unique_keys)
    
    logger.info(f"写入人气排名数据 {count} 条")
    return count


def get_existing_dates(session, start_date: str, end_date: str) -> set:
    """
    获取数据库中已存在的交易日期
    
    Args:
        session: 数据库会话
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        已存在日期的集合
    """
    from sqlalchemy import text
    
    sql = """
        SELECT DISTINCT trade_date 
        FROM stock_popularity_rank 
        WHERE trade_date BETWEEN :start_date AND :end_date
    """
    result = session.execute(text(sql), {"start_date": start_date, "end_date": end_date})
    return {str(row[0]) for row in result.fetchall()}


def get_trading_dates(start_date: str, end_date: str) -> list:
    """
    获取指定范围内的交易日历
    
    Args:
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
    
    Returns:
        list: 交易日列表
    """
    try:
        import qstock as qs
        logger.info(f"正在通过 qstock 获取交易日历：{start_date} 至 {end_date}")
        
        all_dates = pd.date_range(start=start_date, end=end_date)
        trading_dates = []
        
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            try:
                latest = qs.latest_trade_date()
                if latest == date_str:
                    trading_dates.append(date_str)
            except Exception:
                if date.weekday() < 5:
                    trading_dates.append(date_str)
        
        logger.info(f"获取到 {len(trading_dates)} 个交易日")
        return trading_dates
        
    except Exception as e:
        logger.warning(f"获取交易日历失败：{e}，将使用周末判断法")
        all_dates = pd.date_range(start=start_date, end=end_date)
        trading_dates = [d.strftime('%Y-%m-%d') for d in all_dates if d.weekday() < 5]
        return trading_dates


def scan_popularity_rank_for_year(end_date: str = None, save_to_db: bool = False, incremental: bool = True):
    """
    遍历 1 年内每个交易日的人气排名并打印/保存到数据库
    
    Args:
        end_date: 结束日期，默认为今天，格式 'YYYY-MM-DD'
        save_to_db: 是否保存到数据库
        incremental: 是否增量更新（跳过已存在的日期）
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=365)
    start_date = start_dt.strftime('%Y-%m-%d')
    
    logger.info(f"开始扫描人气排名：{start_date} 至 {end_date}")
    
    trading_dates = get_trading_dates(start_date, end_date)
    
    if save_to_db and incremental:
        from app.db import get_session
        with get_session() as session:
            existing_dates = get_existing_dates(session, start_date, end_date)
            logger.info(f"数据库中已存在 {len(existing_dates)} 个日期的数据")
            trading_dates = [d for d in trading_dates if d not in existing_dates]
            logger.info(f"需要获取 {len(trading_dates)} 个新日期的数据")
            
            if not trading_dates:
                logger.info("所有日期的数据都已存在，无需更新")
                return
    
    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(trading_dates)}] 处理日期：{trade_date}")
        logger.info(f"{'='*60}")
        
        df = get_popularity_rank_by_date(trade_date)
        
        if df is not None and not df.empty:
            transformed_df = _transform_rank_data(df, trade_date)
            
            if transformed_df is not None and not transformed_df.empty:
                top_10 = transformed_df[['name', 'rank']].head(10)
                logger.info(f"\n{trade_date} 人气排名前 10:")
                print(top_10.to_string(index=False))
                
                if save_to_db:
                    from app.db import get_session
                    with get_session() as session:
                        count = save_popularity_to_db(transformed_df, session)
                        logger.info(f"已保存 {count} 条数据到数据库")
            else:
                logger.warning(f"{trade_date} 数据转换失败")
        else:
            logger.warning(f"{trade_date} 无数据")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"扫描完成！共处理 {len(trading_dates)} 个交易日")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    # 添加项目根目录到路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    parser = argparse.ArgumentParser(description="股票概念缓存生成器")
    parser.add_argument("--update-cache", action="store_true", help="更新股票概念缓存")
    parser.add_argument("--scan-popularity", action="store_true", help="扫描 1 年内人气排名")
    parser.add_argument("--end-date", type=str, default=None, help="扫描结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--date", type=str, default=None, help="获取指定日期的人气排名，格式 YYYY-MM-DD")
    parser.add_argument("--save-to-db", action="store_true", help="将人气排名数据保存到数据库")
    parser.add_argument("--no-incremental", action="store_true", help="禁用增量更新（全量覆盖）")
    parser.add_argument("--create-table", action="store_true", help="创建数据库表（如果不存在）")
    
    args = parser.parse_args()
    
    if args.create_table:
        from app.db import get_session
        from app.models import get_create_sql
        from sqlalchemy import text
        with get_session() as session:
            sql = get_create_sql("stock_popularity_rank")
            session.execute(text(sql))
            logger.info("已创建表 stock_popularity_rank")
        sys.exit(0)
    
    if args.update_cache:
        update_cache_from_pywencai()
    elif args.scan_popularity:
        scan_popularity_rank_for_year(
            end_date=args.end_date,
            save_to_db=args.save_to_db,
            incremental=not args.no_incremental
        )
    elif args.date:
        df = get_popularity_rank_by_date(args.date)
        if df is not None and not df.empty:
            print(f"\n{args.date} 人气排名前 20:")
            rank_col = next((col for col in df.columns if re.search(r'排名', col)), None)
            name_col = next((col for col in df.columns if '股票简称' in col or '股票名称' in col), None)
            if rank_col and name_col:
                print(df[[name_col, rank_col]].head(20).to_string(index=False))
            
            if args.save_to_db:
                transformed_df = _transform_rank_data(df, args.date)
                if transformed_df is not None and not transformed_df.empty:
                    from app.db import get_session
                    with get_session() as session:
                        count = save_popularity_to_db(transformed_df, session)
                        logger.info(f"已保存 {count} 条数据到数据库")
    else:
        parser.print_help()
