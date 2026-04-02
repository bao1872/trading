# ths_query.py
"""
同花顺数据查询工具

================================================================================
整体架构
================================================================================

【功能模块】
1. 概念/人气查询：--update-cache, --scan-popularity, --date
2. 财务数据查询：--financial-query

【数据库表】
- stock_pools：概念缓存（股票代码、概念列表、人气、市值）
- stock_popularity_rank：人气排名历史数据
- stock_financial_summary：财务汇总数据（YTD 累计值）

================================================================================
财务数据查询流程 (--financial-query)
================================================================================

【数据存储】
- stock_financial_summary 表存储的是 YTD 累计值（原始值）
- 单季度值需要用 get_quarterly_single() 函数计算

【测试模式 (--test)】
Step 6.1: _verify_field_mapping() - loop=False 预检字段映射
Step 6.1.5: 检查日期列存在，空值率
Step 6.2: wc.get(loop=True) 获取首季度数据
Step 6.3: _validate_saved_data() 抽样校验
Step 6.4: 测试模式仅采集首季度，校验通过后才可全量

【全量模式 (--financial-query)】
1. 遍历 all_queries.py 中的所有问句
2. 增量检查（--incremental 时跳过已有数据）
3. 查询数据（loop=True，最多3次重试）
4. 数据清洗（_clean_dataframe_columns）
5. 保存数据库（_save_summary_to_db）
6. 随机等待 10-20秒 防限流

【字段映射机制 (_clean_dataframe_columns)】
1. 从问句提取目标字段列表
2. 模糊匹配 pywencai 返回列（query_to_wencai）
3. 重命名为数据库字段名（db_rename）
4. 固定映射：股票代码→ts_code，股票简称→stock_name
5. 添加 report_date 列

【单季度值计算 (get_quarterly_single)】
- 流量指标：groupby(ts_code).diff() 计算差分
- Q1：保持原 YTD 值（diff() 后为 NaN）
- 时点指标（总资产等）：不处理，直接使用

【关键函数清单】
_check_group_data_exists: 检查数据是否已存在
_extract_expected_date: 从问句提取预期日期
_validate_dataframe: 验证 DataFrame
_load_field_mapping: 加载字段映射 JSON
_clean_dataframe_columns: 清洗列名（匹配+重命名）
_save_summary_to_db: 保存到 stock_financial_summary 表
get_quarterly_single: 计算单季度值（YTD 差分）
_verify_field_mapping: 字段映射预检（loop=False）
_validate_saved_data: 校验保存的数据
execute_financial_queries: 财务查询主入口

================================================================================
Usage
================================================================================
    # 更新股票概念缓存
    python features/ths_query.py --update-cache

    # 获取指定日期的人气排名
    python features/ths_query.py --date 2024-03-01

    # 扫描人气排名（增量更新）
    python features/ths_query.py --scan-popularity

    # 执行财务数据查询（测试模式）
    python features/ths_query.py --financial-query --test

    # 执行财务数据查询（全量）
    python features/ths_query.py --financial-query

    # 增量更新
    python features/ths_query.py --financial-query --incremental

    # 查看帮助
    python features/ths_query.py --help
"""
import pywencai as wc
import pandas as pd
import re
import sys
import os
import logging
import time
import random
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
    industry_col = next((col for col in raw_df.columns if '所属同花顺行业' in col and '二级' not in col), None)
    industry_l2_col = next((col for col in raw_df.columns if '所属同花顺二级行业' in col), None)
    industry_pe_col = next((col for col in raw_df.columns if '行业平均PE' in col or '行业市盈率' in col), None)

    required_cols = [concept_col, rank_col, market_cap_col]
    if not all(required_cols):
        logger.error("在数据源中未能找到所有必需的列（概念、人气、市值）")
        return None

    logger.info(f"成功匹配到列: 概念='{concept_col}', 人气='{rank_col}', 流通市值='{market_cap_col}'")
    logger.info(f"新增列: 总市值='{total_market_cap_col}', 行业三级='{industry_col}', 行业二级='{industry_l2_col}', 行业PE='{industry_pe_col}'")

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
        rename_map[industry_col] = 'industry_l3'
    if industry_l2_col:
        base_cols.append(industry_l2_col)
        rename_map[industry_l2_col] = 'industry_l2'
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
    if 'industry_l2' in cache_df.columns:
        output_cols.append('industry_l2')
    if 'industry_l3' in cache_df.columns:
        output_cols.append('industry_l3')
    if 'industry_pe' in cache_df.columns:
        output_cols.append('industry_pe')
        
    return cache_df[output_cols]

def save_concepts_to_db(df: pd.DataFrame, session) -> int:
    """
    将股票概念缓存数据保存到数据库

    Args:
        df: 数据 DataFrame
        session: 数据库会话

    Returns:
        写入的行数
    """
    from datasource.database import bulk_upsert

    if df is None or df.empty:
        return 0

    unique_keys = ['ts_code']
    count = bulk_upsert(session, 'stock_pools', df, unique_keys)

    logger.info(f"写入股票概念缓存数据 {count} 条")
    return count


def update_cache_from_pywencai(save_to_db: bool = True):
    """
    使用 pywencai 自动获取所有 A 股的概念、人气和市值，并将其保存到数据库。

    Args:
        save_to_db: 是否保存到数据库，默认 True
    """
    logger.info("正在通过问财获取所有 A 股的概念、人气及市值数据...")
    try:
        query_text = '非 st，主板或创业板或科创板，所属概念，人气排名，流通市值，总市值，所属同花顺行业，所属同花顺二级行业，行业平均 PE'
        res = wc.get(query=query_text, loop=True, sleep=2, cookie=None)

        if res is None or res.empty:
            logger.error("未能从问财获取到数据。可能是 Cookie 已过期。")
            return

        logger.info(f"成功获取到 {len(res)} 条原始数据，正在处理...")
        transformed_df = _transform_raw_data(res)

        if transformed_df is not None and not transformed_df.empty:
            if save_to_db:
                logger.info("正在将缓存数据写入数据库...")
                from datasource.database import get_session
                with get_session() as session:
                    count = save_concepts_to_db(transformed_df, session)
                    logger.info(f"成功保存 {count} 条数据到数据库 stock_pools 表")
        else:
            logger.warning("数据转换失败或结果为空。")

    except Exception as e:
        logger.error(f"生成概念缓存时发生错误：{e}")
        raise

def get_popularity_rank_by_date(target_date: str, max_retries: int = 3):
    """
    获取指定日期的股票人气排名数据
    
    Args:
        target_date: 目标日期，格式 'YYYY-MM-DD'
        max_retries: 最大重试次数，默认 3 次
    
    Returns:
        DataFrame: 包含人气排名数据的 DataFrame
    """
    query_text = f'{target_date} 科创板或创业板或主板非st，人气排名，总市值大于200亿'
    
    for attempt in range(1, max_retries + 1):
        import signal
        
        class TimeoutException(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException("wc.get() 调用超时（>10 分钟）")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)
        
        try:
            res = wc.get(query=query_text, loop=True, sleep=2, cookie=None)
            signal.alarm(0)
            
            if res is None:
                logger.warning(f"未能从问财获取到 {target_date} 的数据 (返回 None)")
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                return None
            
            if res.empty:
                logger.warning(f"未能从问财获取到 {target_date} 的数据 (DataFrame 为空)")
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                return None
            
            logger.info(f"成功获取到 {len(res)} 条数据")
            return res
            
        except TimeoutException as e:
            signal.alarm(0)
            logger.error(f"获取 {target_date} 人气排名时超时（第{attempt}次尝试）")
            if attempt < max_retries:
                time.sleep(60)
            else:
                logger.error(f"获取 {target_date} 人气排名失败：已重试 {max_retries} 次")
            continue
            
        except Exception as e:
            signal.alarm(0)
            logger.error(f"获取 {target_date} 人气排名时发生错误：{e}")
            if attempt < max_retries:
                time.sleep(60)
            else:
                logger.error(f"获取 {target_date} 人气排名失败：已重试 {max_retries} 次")
            continue
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
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
    from datasource.database import bulk_upsert
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


def delete_popularity_by_date(session, trade_date: str) -> int:
    """
    删除指定日期的人气排名数据
    
    Args:
        session: 数据库会话
        trade_date: 交易日期，格式 'YYYY-MM-DD'
        
    Returns:
        删除的行数
    """
    from sqlalchemy import text
    
    sql = "DELETE FROM stock_popularity_rank WHERE trade_date = :trade_date"
    result = session.execute(text(sql), {"trade_date": trade_date})
    deleted_count = result.rowcount
    session.commit()
    
    logger.info(f"已删除 {trade_date} 的 {deleted_count} 条数据")
    return deleted_count


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
        
        latest = qs.latest_trade_date()
        logger.info(f"最新交易日：{latest}")
        
        all_dates = pd.date_range(start=start_date, end=end_date)
        trading_dates = []
        
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            if date_str <= latest and date.weekday() < 5:
                trading_dates.append(date_str)
        
        trading_dates.sort()
        logger.info(f"获取到 {len(trading_dates)} 个交易日")
        return trading_dates
        
    except Exception as e:
        logger.warning(f"获取交易日历失败：{e}，将使用周末判断法")
        all_dates = pd.date_range(start=start_date, end=end_date)
        trading_dates = [d.strftime('%Y-%m-%d') for d in all_dates if d.weekday() < 5]
        trading_dates.sort()
        return trading_dates


def scan_popularity_rank_for_year(start_date: str = None, end_date: str = None, save_to_db: bool = False, incremental: bool = True):
    """
    遍历指定日期范围内每个交易日的人气排名并打印/保存到数据库
    
    Args:
        start_date: 开始日期，格式 'YYYY-MM-DD'，默认为 end_date 往前推 365 天
        end_date: 结束日期，默认为今天，格式 'YYYY-MM-DD'
        save_to_db: 是否保存到数据库
        incremental: 是否增量更新（跳过已存在的日期）
    """
    from tqdm import tqdm
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date is None:
        start_dt = end_dt - timedelta(days=365)
        start_date = start_dt.strftime('%Y-%m-%d')
    else:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    
    logger.info(f"开始扫描人气排名：{start_date} 至 {end_date}")
    
    trading_dates = get_trading_dates(start_date, end_date)
    
    if save_to_db and incremental:
        from datasource.database import get_session
        with get_session() as session:
            existing_dates = get_existing_dates(session, start_date, end_date)
            logger.info(f"数据库中已存在 {len(existing_dates)} 个日期的数据")
            trading_dates = [d for d in trading_dates if d not in existing_dates]
            logger.info(f"需要获取 {len(trading_dates)} 个新日期的数据")
            
            if not trading_dates:
                logger.info("所有日期的数据都已存在，无需更新")
                return
    
    pbar = tqdm(trading_dates, desc="获取人气数据", ncols=100)
    for trade_date in pbar:
        pbar.set_postfix_str(trade_date)
        
        df = get_popularity_rank_by_date(trade_date)
        
        if df is not None and not df.empty:
            transformed_df = _transform_rank_data(df, trade_date)
            
            if transformed_df is not None and not transformed_df.empty:
                if save_to_db:
                    from datasource.database import get_session
                    with get_session() as session:
                        count = save_popularity_to_db(transformed_df, session)
                        logger.info(f"{trade_date}: 已保存 {count} 条数据")
        
        sleep_time = random.randint(10, 20)
        time.sleep(sleep_time)
    
    logger.info(f"扫描完成！共处理 {len(trading_dates)} 个交易日")


GROUP_CHECK_FIELDS = {
    "FINANCIAL_SUMMARY": "EBIT",
}


def _check_group_data_exists(report_date: str, group_name: str, session) -> bool:
    """
    检查数据库中该报告期 + 该组字段是否已有数据
    
    Args:
        report_date: 报告期（如 "20210331"）
        group_name: 组名
        session: 数据库会话
    
    Returns:
        True 如果已有数据，False 如果没有
    """
    from sqlalchemy import text
    
    check_field = GROUP_CHECK_FIELDS.get(group_name)
    if not check_field:
        return False
    
    sql = f'''
        SELECT COUNT(*)
        FROM stock_financial_summary
        WHERE report_date = :report_date
          AND "{check_field}" IS NOT NULL
    '''
    result = session.execute(text(sql), {'report_date': report_date})
    count = result.scalar()
    return count > 0


def _extract_expected_date(query: str) -> str | None:
    """从问句中提取预期的日期（格式：YYYY年X季度 -> YYYYMMDD）"""
    import re
    match = re.search(r'(\d{4})年(一季度|二季度|三季度|四季度)', query)
    if not match:
        return None
    
    year = match.group(1)
    quarter = match.group(2)
    
    quarter_end_dates = {
        "一季度": "0331",
        "二季度": "0630",
        "三季度": "0930",
        "四季度": "1231",
    }
    
    return year + quarter_end_dates[quarter]


def _validate_dataframe(df: pd.DataFrame, expected_date: str, min_rows: int = 100) -> tuple[bool, str]:
    """
    验证 DataFrame 是否符合预期
    
    Args:
        df: 待验证的 DataFrame
        expected_date: 预期的日期后缀（如 "20210331"）
        min_rows: 最小行数要求
    
    Returns:
        (是否有效, 错误信息)
    """
    if df is None or df.empty:
        return False, "DataFrame 为空"
    
    if len(df) < min_rows:
        return False, f"数据行数过少 ({len(df)} < {min_rows})，可能查询被误解"
    
    date_cols = [col for col in df.columns if f'[{expected_date}]' in col]
    if not date_cols:
        return False, f"未找到预期日期 [{expected_date}] 的列，可能时间不匹配"
    
    invalid_keywords = ['溯源', '文章', '相关产品']
    for kw in invalid_keywords:
        if any(kw in col for col in df.columns):
            return False, f"包含无关字段（{kw}），问句可能被误解"
    
    return True, ""


def _load_field_mapping():
    """从 JSON 文件加载字段映射关系"""
    import json
    mapping_file = os.path.join(os.path.dirname(__file__), '..', 'pywencai_queries', 'field_mapping.json')
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _clean_dataframe_columns(df: pd.DataFrame, query: str, expected_date: str) -> pd.DataFrame:
    """
    清洗 DataFrame 列：
    1. 从 JSON 加载映射，匹配 pywencai 返回列
    2. 去掉日期后缀，重命名为用户字段名
    3. 添加 report_date
    """
    import re

    match = re.search(r'的(.+)$', query)
    if not match:
        return df

    target_fields = [f.strip() for f in match.group(1).split('、')]

    mapping = _load_field_mapping()
    query_to_wencai = mapping['query_to_wencai']
    db_rename = mapping.get('db_rename', {})

    rename_map = {}
    for user_field in target_fields:
        wencai_col = query_to_wencai.get(user_field, user_field)
        matched_col = None
        for col in df.columns:
            wencai_col_stripped = wencai_col.split('[')[0].strip()
            if wencai_col_stripped in col:
                matched_col = col
                break
        if matched_col:
            db_field = db_rename.get(wencai_col, wencai_col)
            rename_map[matched_col] = db_field

    rename_map['股票代码'] = 'ts_code'
    rename_map['股票简称'] = 'stock_name'

    renamed_df = df[list(rename_map.keys())].rename(columns=rename_map)

    result_df = renamed_df.copy()
    result_df['report_date'] = expected_date

    return result_df


def _save_summary_to_db(df: pd.DataFrame, session) -> int:
    """
    将财务汇总数据保存到 stock_financial_summary 表

    Args:
        df: 清洗后的 DataFrame
        session: 数据库会话

    Returns:
        写入的行数
    """
    from sqlalchemy import text

    if df.empty:
        return 0

    db_columns = [
        'ts_code', 'stock_name', 'report_date',
        '营业总收入', '营业收入', '营业成本', '营业利润',
        '归母净利润', '少数股东损益',
        'EBIT', '经营活动现金流净额',
        '资本开支', '销售商品提供劳务收到的现金',
        '总资产', '应收账款', '存货', '应付账款',
        '合同负债', 'EBITDA', '研发费用',
        '期末现金', '股东权益',
    ]

    col_clause = ', '.join(db_columns)
    placeholders = ', '.join([f':{c}' for c in db_columns])
    update_cols = [c for c in db_columns if c not in ['ts_code', 'report_date']]
    update_clause = ', '.join([f'{c} = COALESCE(EXCLUDED.{c}, stock_financial_summary.{c})' for c in update_cols])

    sql = f"""
        INSERT INTO stock_financial_summary ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT (ts_code, report_date) DO UPDATE SET
            {update_clause},
            updated_at = CURRENT_TIMESTAMP
    """

    records = []
    for _, row in df.iterrows():
        record = {
            'ts_code': row.get('ts_code'),
            'stock_name': row.get('stock_name'),
            'report_date': row.get('report_date'),
        }

        for col in db_columns[3:]:
            val = row.get(col)
            if pd.isna(val):
                record[col] = None
            else:
                record[col] = float(val) if isinstance(val, (int, float)) else val

        records.append(record)

    for record in records:
        session.execute(text(sql), record)

    logger.info(f"写入财务汇总数据 {len(records)} 条到数据库 stock_financial_summary 表")
    return len(records)


def get_quarterly_single(df: pd.DataFrame, ts_code: str = None) -> pd.DataFrame:
    """
    计算单季度财务数据（YTD 差分）

    数据库 stock_financial_summary 表存储的是 YTD 累计值，
    使用此函数计算单季度值（流量指标）。

    Args:
        df: 包含 YTD 累计值的 DataFrame，需包含 ts_code, report_date 字段
        ts_code: 可选，筛选特定股票

    Returns:
        DataFrame：单季度值
        - 流量指标：通过 diff() 计算差分（本期 - 上期）
        - Q1：保持原 YTD 值（diff() 后为 NaN）
        - 时点指标（总资产、应收账款等）：不处理，直接使用

    Example:
        from features.ths_query import get_quarterly_single

        # 从 DB 读取 YTD 数据
        df = pd.read_sql("SELECT * FROM stock_financial_summary WHERE ...", conn)

        # 计算单季度值
        df_single = get_quarterly_single(df, ts_code='000657.SZ')
    """
    flow_fields = [
        '营业总收入', '营业收入', '营业成本', '营业利润',
        '归母净利润', '少数股东损益', 'EBIT',
        '经营活动现金流净额', '资本开支', '销售商品提供劳务收到的现金',
        '研发费用'
    ]

    if ts_code:
        df = df[df['ts_code'] == ts_code].copy()

    result = df.copy()
    for field in flow_fields:
        if field in result.columns:
            result[field] = result.groupby('ts_code')[field].diff()

    for field in flow_fields:
        if field in result.columns:
            mask = result[field].isna()
            result.loc[mask, field] = df.loc[mask, field]

    return result


def _verify_field_mapping(query: str, expected_date: str) -> tuple[bool, str, list]:
    """
    用 loop=False 预检字段映射，返回 (是否成功, 错误信息, 实际列名列表)
    """
    try:
        res = wc.get(query=query, loop=False, sleep=2, cookie=None)
        if res is None or res.empty:
            return False, "预检返回空数据", []
        cols = res.columns.tolist()
        return True, "", cols
    except Exception as e:
        return False, f"预检异常: {e}", []


def _validate_saved_data(report_date: str, session, sample_count: int = 5) -> tuple[bool, str]:
    """
    从 stock_financial_summary 抽样校验关键字段是否非空且量级合理
    返回 (是否通过, 错误信息)
    """
    from sqlalchemy import text
    import random

    check_fields = ['归母净利润', 'EBIT', '总资产', '应收账款']
    field_checks = ' AND '.join([f'"{f}" IS NOT NULL' for f in check_fields])

    sql = f"""
        SELECT ts_code, stock_name, 归母净利润, EBIT, 总资产, 应收账款
        FROM stock_financial_summary
        WHERE report_date = :report_date AND {field_checks}
    """
    result = session.execute(text(sql), {'report_date': report_date})
    rows = result.fetchall()

    if len(rows) < sample_count:
        return False, f"有效数据行数不足 ({len(rows)} < {sample_count})"

    sample = random.sample(rows, min(sample_count, len(rows)))
    issues = []
    for row in sample:
        ts_code, name, net_profit, ebit, total_assets, receivables = row
        if abs(net_profit) < 100 or abs(ebit) < 100:
            issues.append(f"{ts_code} ({name}): 归母净利润={net_profit}, EBIT={ebit} 量级异常")
        if total_assets < 1e8:
            issues.append(f"{ts_code} ({name}): 总资产={total_assets} 量级异常")

    if issues:
        return False, "数据校验失败:\n  " + "\n  ".join(issues)
    return True, ""


def execute_financial_queries(group_name: str = None, test_mode: bool = False, save_to_db: bool = True, incremental: bool = False):
    """
    执行财务汇总数据查询并保存到 stock_financial_summary 表

    Args:
        group_name: 指定要查询的组名（仅支持 FINANCIAL_SUMMARY）
        test_mode: 测试模式，先字段预检再获取首季度，校验通过后才可全量
        save_to_db: 是否保存到数据库，默认 True
        incremental: 增量更新模式，跳过已有数据的报告期
    """
    from tqdm import tqdm
    import sys

    from pywencai_queries.all_queries import ALL_QUERIES, START_YEAR, YEAR_LIST, REPORT_PERIOD_LIST

    groups_to_query = ALL_QUERIES

    logger.info(f"起始年份: {START_YEAR}")
    logger.info(f"年份列表: {YEAR_LIST}")
    logger.info(f"报告期数量: {len(REPORT_PERIOD_LIST)}")

    first_query = groups_to_query["FINANCIAL_SUMMARY"][0]
    expected_date = _extract_expected_date(first_query)
    logger.info(f"首问句: {first_query[:60]}...")
    logger.info(f"预期日期: {expected_date}")

    if test_mode:
        logger.info("\n========== Step 6.1: 字段映射预检 (loop=False) ==========")
        ok, err, cols = _verify_field_mapping(first_query, expected_date)
        if not ok:
            logger.error(f"字段映射预检失败: {err}")
            logger.error("请检查问句字段名是否与同花顺返回列名匹配，中断执行。")
            return

        logger.info(f"预检成功，返回列数: {len(cols)}")
        wencai_values_to_check = list(_load_field_mapping()['query_to_wencai'].values())
        missing = [v for v in wencai_values_to_check if not any(v.split('[')[0].strip() in c for c in cols)]
        if missing:
            logger.error(f"字段映射缺失: {missing}")
            logger.error("请检查问句字段名是否正确，中断执行。")
            return
        logger.info("字段映射预检通过!")

        logger.info("\n========== Step 6.1.5: 数据正确性与完整性检查 ==========")
        res_check = wc.get(query=first_query, loop=False, sleep=2, cookie=None)
        if res_check is None or res_check.empty:
            logger.error("数据正确性检查返回空数据")
            return
        check_fields_map = {
            '营业总收入': '营业总收入', '营业收入': '营业收入', '营业成本': '营业成本',
            '营业利润': '营业利润', '归母净利润': '归属于母公司所有者的净利润',
            '少数股东损益': '少数股东损益', 'EBIT': '息税前利润ebit',
            '经营活动现金流净额': '经营活动产生的现金流量净额',
            '资本开支': '购建固定资产、有形资产和其他长期资产支付的现金',
            '销售商品提供劳务收到的现金': '销售商品、提供劳务收到的现金',
            '总资产': '资产总计', '应收账款': '应收账款', '存货': '存货',
            '应付账款': '应付票据及应付账款', '合同负债': '合同负债',
            'EBITDA': '息税折旧摊销前利润ebitda', '研发费用': '研发费用',
            '期末现金': '期末现金及现金等价物余额', '股东权益': '股东权益合计',
        }
        date_col_found = next((c for c in res_check.columns if f'[{expected_date}]' in c), None)
        if not date_col_found:
            logger.error(f"未找到日期列 [{expected_date}]，数据可能被误解")
            return
        logger.info(f"日期列存在: {date_col_found}")
        row_count = len(res_check)
        logger.info(f"预检数据行数: {row_count}")
        if row_count < 100:
            logger.error(f"数据行数过少 ({row_count} < 100)，问句可能被误解")
            return
        null_counts = {}
        for db_field, col_pattern in check_fields_map.items():
            matched = next((c for c in res_check.columns if col_pattern in c), None)
            if matched:
                null_pct = res_check[matched].isna().mean() * 100
                null_counts[db_field] = f"{null_pct:.1f}%"
        logger.info("各字段空值率:")
        for f, pct in null_counts.items():
            logger.info(f"  {f}: {pct}")
        high_null = [f for f, pct in null_counts.items() if float(pct.rstrip('%')) > 30]
        if high_null:
            logger.warning(f"以下字段空值率 > 30%: {high_null}")
        logger.info("数据正确性与完整性检查通过!")

        logger.info("\n========== Step 6.2: 首季度采集 (loop=True) ==========")
        res = wc.get(query=first_query, loop=True, sleep=2, cookie=None)
        if res is None or res.empty:
            logger.error("首季度数据获取失败")
            return

        is_valid, err_msg = _validate_dataframe(res, expected_date)
        if not is_valid:
            logger.error(f"数据验证失败: {err_msg}")
            return

        res = _clean_dataframe_columns(res, first_query, expected_date)
        res['查询问句'] = first_query

        if save_to_db:
            from datasource.database import get_session
            with get_session() as session:
                count = _save_summary_to_db(res, session)
                logger.info(f"首季度数据库保存: {count} 条记录")

                logger.info("\n========== Step 6.3: 数据校验 ==========")
                ok, err = _validate_saved_data(expected_date, session)
                if not ok:
                    logger.error(f"数据校验失败:\n  {err}")
                    logger.error("中断执行，请检查问句和数据。")
                    return
                logger.info("数据校验通过!")

        all_results = [res]
        logger.info("\n========== Step 6.4: 测试模式仅采集首季度 ==========")
        logger.info(f"首季度 {expected_date} 已完成，退出（test 模式）。")
    else:
        all_results = []
        queries_to_run = groups_to_query["FINANCIAL_SUMMARY"]

        pbar = tqdm(queries_to_run, desc="查询 FINANCIAL_SUMMARY", ncols=100)
        for idx, query in enumerate(pbar):
            pbar.set_postfix_str(f"第{idx+1}/{len(queries_to_run)}条")

            q_expected_date = _extract_expected_date(query)

            if incremental and q_expected_date:
                from datasource.database import get_session
                with get_session() as session:
                    if _check_group_data_exists(q_expected_date, "FINANCIAL_SUMMARY", session):
                        logger.info(f"  [{idx+1}/{len(queries_to_run)}] 跳过（已有数据）: {q_expected_date}")
                        continue

            max_retries = 3
            retry_count = 0
            success = False

            while retry_count <= max_retries and not success:
                if retry_count > 0:
                    logger.info(f"  [{idx+1}/{len(queries_to_run)}] 第 {retry_count} 次重试，等待 60 秒...")
                    time.sleep(60)

                try:
                    res = wc.get(query=query, loop=True, sleep=2, cookie=None)

                    if res is not None and not res.empty:
                        if q_expected_date:
                            is_valid, err_msg = _validate_dataframe(res, q_expected_date)
                            if not is_valid:
                                logger.warning(f"  [{idx+1}/{len(queries_to_run)}] 数据验证失败: {err_msg}")
                                retry_count += 1
                                continue

                            res = _clean_dataframe_columns(res, query, q_expected_date)

                        res['查询问句'] = query
                        all_results.append(res)
                        logger.info(f"  [{idx+1}/{len(queries_to_run)}] 成功获取 {len(res)} 条数据")

                        if save_to_db:
                            from datasource.database import get_session
                            with get_session() as session:
                                count = _save_summary_to_db(res, session)
                                logger.info(f"  [{idx+1}/{len(queries_to_run)}] 数据库保存: {count} 条记录")

                        success = True
                    else:
                        logger.warning(f"  [{idx+1}/{len(queries_to_run)}] 未获取到数据")
                        retry_count += 1

                except Exception as e:
                    logger.error(f"  [{idx+1}/{len(queries_to_run)}] 查询失败: {e}")
                    retry_count += 1

            if not success:
                logger.error(f"  [{idx+1}/{len(queries_to_run)}] 重试 {max_retries} 次后仍失败，跳过")

            if success:
                sleep_time = random.randint(10, 20)
                time.sleep(sleep_time)

        logger.info(f"查询完成！共处理 {len(all_results)} 个报告期")


if __name__ == '__main__':
    import argparse
    
    # 添加项目根目录到路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    parser = argparse.ArgumentParser(description="同花顺数据查询工具")
    parser.add_argument("--update-cache", action="store_true", help="更新股票概念缓存（保存到数据库）")
    parser.add_argument("--no-db", action="store_true", help="不保存到数据库（调试用）")
    parser.add_argument("--scan-popularity", action="store_true", help="扫描指定日期范围内人气排名")
    parser.add_argument("--start-date", type=str, default=None, help="扫描开始日期，格式 YYYY-MM-DD，默认为结束日期往前推 1 年")
    parser.add_argument("--end-date", type=str, default=None, help="扫描结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--date", type=str, default=None, help="获取指定日期的人气排名，格式 YYYY-MM-DD")
    parser.add_argument("--save-to-db", action="store_true", help="将人气排名数据保存到数据库")
    parser.add_argument("--no-incremental", action="store_true", help="禁用增量更新（全量覆盖）")
    parser.add_argument("--force-update-date", type=str, default=None, help="强制更新指定日期的数据（先删除再插入）")
    parser.add_argument("--create-table", action="store_true", help="创建数据库表（如果不存在）")
    parser.add_argument("--financial-query", action="store_true", help="执行财务数据查询并保存到数据库")
    parser.add_argument("--group", type=str, default=None, help="指定财务查询的组名（配合 --financial-query 使用）")
    parser.add_argument("--test", action="store_true", help="测试模式，每组只查询第一条问句")
    parser.add_argument("--incremental", action="store_true", help="增量更新模式，跳过已有数据的报告期")
    
    args = parser.parse_args()
    
    if args.create_table:
        from datasource.database import get_session
        from app.models import get_create_sql
        from sqlalchemy import text
        with get_session() as session:
            # 创建人气排名表
            sql = get_create_sql("stock_popularity_rank")
            session.execute(text(sql))
            logger.info("已创建表 stock_popularity_rank")
            
            # 创建股票概念缓存表
            sql = get_create_sql("stock_pools")
            session.execute(text(sql))
            logger.info("已创建表 stock_pools")
            
            # 创建财务数据表
            sql = get_create_sql("stock_financial_data")
            session.execute(text(sql))
            logger.info("已创建表 stock_financial_data")

            # 创建财务汇总表
            sql = get_create_sql("stock_financial_summary")
            session.execute(text(sql))
            logger.info("已创建表 stock_financial_summary")
        sys.exit(0)
    
    if args.update_cache:
        save_to_db = not args.no_db
        update_cache_from_pywencai(save_to_db=save_to_db)
    elif args.force_update_date:
        logger.info(f"强制更新 {args.force_update_date} 的数据...")
        from datasource.database import get_session
        with get_session() as session:
            delete_popularity_by_date(session, args.force_update_date)
        
        df = get_popularity_rank_by_date(args.force_update_date)
        if df is not None and not df.empty:
            transformed_df = _transform_rank_data(df, args.force_update_date)
            if transformed_df is not None and not transformed_df.empty:
                from datasource.database import get_session
                with get_session() as session:
                    count = save_popularity_to_db(transformed_df, session)
                    logger.info(f"强制更新完成，已保存 {count} 条数据")
            else:
                logger.warning(f"数据转换失败，未写入数据库")
        else:
            logger.warning(f"未能获取到 {args.force_update_date} 的数据")
    elif args.financial_query:
        execute_financial_queries(group_name=args.group, test_mode=args.test, save_to_db=not args.no_db, incremental=args.incremental)
    elif args.scan_popularity:
        scan_popularity_rank_for_year(
            start_date=args.start_date,
            end_date=args.end_date,
            save_to_db=True,
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
                    from datasource.database import get_session
                    with get_session() as session:
                        count = save_popularity_to_db(transformed_df, session)
                        logger.info(f"已保存 {count} 条数据到数据库")
    else:
        parser.print_help()
