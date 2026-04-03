"""
Purpose: 日线滚动窗口放量异动统计
- 以5个交易日为窗口resample日线数据
- 每个窗口计算总成交量
- 用过去51个窗口计算zscore

Usage:
    python signals_abnormal.py --date 2026-03-20 --top-themes 21 --top-stocks 200 --top-concepts 30
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from utils.concept_mapping import load_concept_data, build_stock_to_concepts, build_concept_to_stocks
from utils.theme_aggregation import (
    load_theme_mapping, build_concept_to_theme_map, get_excluded_concepts,
    aggregate_theme_with_concept_detail
)
from datasource.k_data_loader import iter_k_data_with_names, load_all_stock_data
from datasource.database import get_session, bulk_upsert, execute_sql


def calculate_5day_rolling_zscore_vectorized(
    all_data: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    window: int = 51
) -> pd.DataFrame:
    """
    向量化计算所有股票的5日滚动成交量Z-Score

    Returns:
        DataFrame: {code, name, zscore, volume, rolling_mean, date}
    """
    all_data = all_data.copy()
    all_data['volume_5d'] = all_data.groupby('ts_code')['volume'].transform(
        lambda x: x.rolling(window=5).sum()
    )
    all_data['rolling_mean'] = all_data.groupby('ts_code')['volume_5d'].transform(
        lambda x: x.rolling(window=window).mean()
    )
    all_data['rolling_std'] = all_data.groupby('ts_code')['volume_5d'].transform(
        lambda x: x.rolling(window=window).std()
    )
    all_data['zscore'] = (all_data['volume_5d'] - all_data['rolling_mean']) / all_data['rolling_std']

    all_data['close_prev'] = all_data.groupby('ts_code')['close'].shift(1)
    all_data['price_change'] = (all_data['close'] - all_data['close_prev']) / all_data['close_prev'] * 100

    snapshot_start = snapshot_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    snapshot_end = snapshot_ts.replace(hour=23, minute=59, second=59, microsecond=999999)
    mask = (all_data['bar_time'] >= snapshot_start) & (all_data['bar_time'] <= snapshot_end)
    snapshot_data = all_data[mask].copy()

    results = []
    for _, row in snapshot_data.iterrows():
        results.append({
            'code': row['ts_code'],
            'name': row['name'],
            'zscore': row['zscore'],
            'volume': row['volume'],
            'rolling_mean': row['rolling_mean'],
            'date': row['bar_time'],
            'price_change': round(row['price_change'], 2) if pd.notna(row['price_change']) else None,
        })
    return pd.DataFrame(results)


def ensure_signal_tables(session) -> None:
    """确保信号表存在，不存在则创建（PostgreSQL）"""
    from datasource.database import execute_sql

    tables_sql = {
        'theme_signals_rolling': """
            CREATE TABLE IF NOT EXISTS theme_signals_rolling (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                period VARCHAR(20),
                theme VARCHAR(100) NOT NULL,
                total_zscore FLOAT,
                avg_zscore FLOAT,
                stock_count INTEGER,
                strength FLOAT,
                anomalous_concept_count INTEGER,
                total_concept_count INTEGER,
                concept_coverage FLOAT,
                top_stocks TEXT,
                top_concepts TEXT,
                UNIQUE(snapshot_date, theme)
            )
        """,
        'concept_signals_rolling': """
            CREATE TABLE IF NOT EXISTS concept_signals_rolling (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                concept VARCHAR(100) NOT NULL,
                theme VARCHAR(100),
                total_zscore FLOAT,
                avg_zscore FLOAT,
                anomalous_stock_count INTEGER,
                total_stock_count INTEGER,
                normalized_strength FLOAT,
                intensity FLOAT,
                top_stocks TEXT,
                UNIQUE(snapshot_date, concept)
            )
        """,
        'stock_anomaly_signals_rolling': """
            CREATE TABLE IF NOT EXISTS stock_anomaly_signals_rolling (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                code VARCHAR(20) NOT NULL,
                name VARCHAR(50),
                zscore FLOAT,
                price_change FLOAT,
                volume_cv FLOAT,
                volume_spearman FLOAT,
                date DATE,
                themes_str TEXT,
                concepts_str TEXT,
                UNIQUE(snapshot_date, code)
            )
        """
    }
    for table_name, sql in tables_sql.items():
        execute_sql(sql)


def save_signals_to_database(signals: dict) -> None:
    """保存异动信号结果到数据库"""
    import pandas as pd

    snapshot_date = signals['snapshot_date']

    with get_session() as session:
        ensure_signal_tables(session)

        theme_df = signals['theme_ranking']
        if not theme_df.empty:
            theme_df = theme_df.copy()
            theme_df['snapshot_date'] = snapshot_date
            if 'top_stocks' in theme_df.columns:
                theme_df['top_stocks'] = theme_df['top_stocks'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else x
                )
            theme_cols = ['snapshot_date', 'theme', 'total_zscore', 'avg_zscore', 'stock_count',
                         'strength', 'anomalous_concept_count', 'total_concept_count',
                         'concept_coverage', 'top_stocks', 'top_concepts']
            theme_cols = [c for c in theme_cols if c in theme_df.columns]
            bulk_upsert(session, 'theme_signals_rolling', theme_df[theme_cols], ['snapshot_date', 'theme'])

        concept_df = signals['concept_ranking']
        if not concept_df.empty:
            concept_df = concept_df.copy()
            concept_df['snapshot_date'] = snapshot_date
            if 'top_stocks' in concept_df.columns:
                concept_df['top_stocks'] = concept_df['top_stocks'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else x
                )
            concept_cols = ['snapshot_date', 'concept', 'theme', 'total_zscore', 'avg_zscore',
                          'anomalous_stock_count', 'total_stock_count', 'normalized_strength',
                          'intensity', 'top_stocks']
            concept_cols = [c for c in concept_cols if c in concept_df.columns]
            bulk_upsert(session, 'concept_signals_rolling', concept_df[concept_cols], ['snapshot_date', 'concept'])

        stock_df = signals['stock_anomaly']
        if not stock_df.empty:
            stock_df = stock_df.copy()
            stock_df['snapshot_date'] = snapshot_date
            if 'date' in stock_df.columns:
                stock_df['date'] = stock_df['date'].apply(lambda x: str(x)[:10] if hasattr(x, 'strftime') else x)
            stock_cols = ['snapshot_date', 'code', 'name', 'zscore', 'price_change', 'volume_cv',
                         'volume_spearman', 'date', 'themes_str', 'concepts_str']
            stock_cols = [c for c in stock_cols if c in stock_df.columns]
            bulk_upsert(session, 'stock_anomaly_signals_rolling', stock_df[stock_cols], ['snapshot_date', 'code'])


def calculate_volume_cv_and_spearman(
    freq: str,
    codes: list,
    snapshot_ts: pd.Timestamp
) -> dict:
    """
    批量计算CV和Spearman

    Returns:
        dict: {code: (cv, spearman)}
    """
    code_set = set(codes)
    result = {code: (None, None) for code in codes}

    start_date = (snapshot_ts - pd.Timedelta(days=10)).strftime('%Y-%m-%d')

    for code, name, df in iter_k_data_with_names(freq=freq, start_date=start_date):
        if code not in code_set:
            continue
        if 'volume' not in df.columns:
            continue

        window_start = snapshot_ts - pd.Timedelta(days=6)
        window_end = snapshot_ts + pd.Timedelta(days=1)

        # 确保时区一致：将 window_start/end 转换为与 df.index 相同的时区
        if df.index.tz is not None:
            window_start = window_start.tz_localize(df.index.tz) if window_start.tz is None else window_start.tz_convert(df.index.tz)
            window_end = window_end.tz_localize(df.index.tz) if window_end.tz is None else window_end.tz_convert(df.index.tz)

        mask = (df.index >= window_start) & (df.index <= window_end)
        window_volumes = df.loc[mask, 'volume'].dropna()
        window_volumes = window_volumes[window_volumes > 0]

        if len(window_volumes) < 2:
            continue

        cv = window_volumes.std(ddof=1) / window_volumes.mean()
        cv = round(cv, 3) if not np.isnan(cv) else None

        days = np.arange(len(window_volumes))
        rho, _ = spearmanr(days, window_volumes.values)
        rho = round(rho, 3) if not np.isnan(rho) else None

        result[code] = (cv, rho)

    return result


def generate_rolling_signals(
    freq: str = 'daily',
    snapshot_date: str = None,
    zscore_threshold: float = 2.0,
    top_n_themes: int = 10,
    top_n_stocks: int = 20,
    top_n_concepts: int = 20,
    limit_stocks: int = None,
    preloaded_data: tuple = None
) -> dict:
    """
    生成滚动窗口异动信号

    Args:
        preloaded_data: 可选，(DataFrame, name_map) 元组，若传入则直接使用，不重复加载
    """
    snapshot_ts = pd.Timestamp(snapshot_date) if snapshot_date else pd.Timestamp.now()

    if preloaded_data is not None:
        all_data, name_map = preloaded_data
        all_data = all_data.copy()
    else:
        start_date = (snapshot_ts - pd.Timedelta(days=51 + 60)).strftime('%Y-%m-%d')
        print("加载全量股票数据...")
        all_data, name_map = load_all_stock_data(freq, start_date=start_date, limit_stocks=limit_stocks)

    if all_data.empty:
        return {
            'snapshot_date': snapshot_date,
            'theme_ranking': pd.DataFrame(),
            'concept_ranking': pd.DataFrame(),
            'stock_anomaly': pd.DataFrame(),
            'cross_signals': pd.DataFrame()
        }

    print(f"  数据量: {len(all_data)} 行, {len(name_map)} 只股票")
    print("  计算5日窗口Z-Score (向量化)...")
    volume_df = calculate_5day_rolling_zscore_vectorized(all_data, snapshot_ts, window=51)

    if volume_df.empty:
        return {
            'snapshot_date': snapshot_date,
            'theme_ranking': pd.DataFrame(),
            'concept_ranking': pd.DataFrame(),
            'stock_anomaly': pd.DataFrame(),
            'cross_signals': pd.DataFrame()
        }

    volume_df = volume_df.sort_values('zscore', ascending=False)

    print("加载概念和主题映射...")
    concept_df = load_concept_data()
    s2c = build_stock_to_concepts(concept_df)
    c2s = build_concept_to_stocks(concept_df)

    print("聚合主题和概念...")
    theme_df, concept_ranking_df, detail_df = aggregate_theme_with_concept_detail(
        volume_df, s2c, c2s, zscore_threshold
    )

    if not theme_df.empty and not detail_df.empty:
        top_concepts = (
            detail_df[detail_df['is_anomalous'] == True]
            .sort_values('total_zscore', ascending=False)
            .groupby('theme')
            .head(3)
            .groupby('theme')['concept']
            .apply(lambda x: ','.join(x))
            .reset_index()
            .rename(columns={'concept': 'top_concepts'})
        )
        theme_df = theme_df.merge(top_concepts, on='theme', how='left')

    anomalous_df = volume_df[volume_df['zscore'] >= zscore_threshold].copy()

    theme_mapping = load_theme_mapping()
    concept_to_theme = build_concept_to_theme_map(theme_mapping)
    excluded = get_excluded_concepts(theme_mapping)

    if not anomalous_df.empty and bool(theme_mapping):
        anomalous_df['themes'] = anomalous_df['code'].map(
            lambda x: [concept_to_theme.get(c, '其他') for c in s2c.get(x, []) if c in concept_to_theme and c not in excluded]
        )
        anomalous_df['themes'] = anomalous_df['themes'].map(lambda x: list(set(x)) if x else [])
        anomalous_df['themes_str'] = anomalous_df['themes'].map(lambda x: ','.join(x[:3]) if x else '其他')
        anomalous_df['concepts_str'] = anomalous_df['code'].map(
            lambda x: ','.join(s2c.get(x, [])) if s2c.get(x) else ''
        )
        anomalous_df = anomalous_df[anomalous_df['themes'].map(lambda x: len(x) > 0)]

    stock_anomaly = anomalous_df.copy()

    if not stock_anomaly.empty:
        snapshot_ts = pd.Timestamp(snapshot_date) if snapshot_date else pd.Timestamp.now()
        print(f"计算CV和Spearman ({len(stock_anomaly)} 只股票)...")
        cv_sp_map = calculate_volume_cv_and_spearman(
            freq, stock_anomaly['code'].tolist(), snapshot_ts
        )
        stock_anomaly['volume_cv'] = stock_anomaly['code'].map(lambda x: cv_sp_map.get(x, (None, None))[0])
        stock_anomaly['volume_spearman'] = stock_anomaly['code'].map(lambda x: cv_sp_map.get(x, (None, None))[1])

    code_to_name = volume_df[['code', 'name']].drop_duplicates().set_index('code')['name'].to_dict()

    def replace_stocks_with_names(df):
        if df.empty or 'top_stocks' not in df.columns:
            return df
        df = df.copy()
        df['top_stocks'] = df['top_stocks'].apply(
            lambda stocks: [code_to_name.get(s, s) for s in stocks] if isinstance(stocks, list) else stocks
        )
        return df

    theme_ranking_df = replace_stocks_with_names(theme_df if not theme_df.empty else pd.DataFrame())
    concept_ranking_df = replace_stocks_with_names(concept_ranking_df)

    return {
        'snapshot_date': snapshot_date,
        'theme_ranking': theme_ranking_df,
        'concept_ranking': concept_ranking_df,
        'stock_anomaly': stock_anomaly if not stock_anomaly.empty else pd.DataFrame(),
        'cross_signals': pd.DataFrame()
    }


def export_signals_excel(signals: dict, output_dir: str = '../review') -> str:
    """导出信号报告为Excel文件"""
    import os

    snapshot_date = signals['snapshot_date']
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'signal_{snapshot_date}.xlsx')

    column_rename_theme = {
        'theme': '主题', 'total_zscore': '总Z分数', 'avg_zscore': '平均Z分数',
        'stock_count': '股票数量', 'strength': '强度', 'top_stocks': 'TOP股票',
        'anomalous_concept_count': '异动概念数', 'total_concept_count': '概念总数',
        'concept_coverage': '概念覆盖率', 'top_concepts': '贡献概念',
    }

    column_rename_concept = {
        'concept': '概念', 'theme': '所属主题', 'total_zscore': '总Z分数',
        'avg_zscore': '平均Z分数', 'anomalous_stock_count': '异动股数',
        'total_stock_count': '成分股总数', 'normalized_strength': '归一化强度',
        'intensity': '集中度', 'top_stocks': 'TOP股票',
    }

    column_rename_stock = {
        'code': '代码', 'name': '名称', 'zscore': 'Z分数',
        'price_change': '涨跌幅(%)',
        'volume_cv': '成交量均匀度(CV)', 'volume_spearman': '放量顺序(ρ)',
        'date': '日期', 'themes': '主题列表', 'themes_str': '主题',
        'concepts_str': '概念'
    }

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        has_data = False

        if not signals['theme_ranking'].empty:
            signals['theme_ranking'].rename(columns=column_rename_theme).to_excel(writer, sheet_name='主题排名', index=False)
            has_data = True

        if not signals['concept_ranking'].empty:
            signals['concept_ranking'].rename(columns=column_rename_concept).to_excel(writer, sheet_name='概念排名', index=False)
            has_data = True

        if not signals['stock_anomaly'].empty:
            cols = ['code', 'name', 'zscore', 'price_change', 'volume_cv', 'volume_spearman', 'date', 'themes', 'themes_str']
            if 'concepts_str' in signals['stock_anomaly'].columns:
                cols.append('concepts_str')
            df_stock = signals['stock_anomaly'][cols].copy()
            df_stock = df_stock.rename(columns=column_rename_stock)
            df_stock.to_excel(writer, sheet_name='个股异动', index=False)
            has_data = True

        if not signals['cross_signals'].empty:
            signals['cross_signals'].to_excel(writer, sheet_name='交叉信号', index=False)
            has_data = True

        if not has_data:
            pd.DataFrame({'message': ['No data available']}).to_excel(writer, sheet_name='无数据', index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except Exception:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='日线滚动窗口异动统计（5日窗口resample）')
    parser.add_argument('--date', type=str, default='latest', help='周截面日期')
    parser.add_argument('--zscore', type=float, default=2.0, help='Z-Score阈值')
    parser.add_argument('--top-themes', type=int, default=21, help='TOP主题数')
    parser.add_argument('--top-stocks', type=int, default=200, help='TOP股票数')
    parser.add_argument('--top-concepts', type=int, default=30, help='TOP概念数')
    parser.add_argument('--db', action='store_true', help='输出到数据库')
    parser.add_argument('--output', type=str, default='../review', help='输出目录')
    args = parser.parse_args()

    snapshot_date = None if args.date == 'latest' else args.date

    signals = generate_rolling_signals(
        freq='daily',
        snapshot_date=snapshot_date,
        zscore_threshold=args.zscore,
        top_n_themes=args.top_themes,
        top_n_stocks=args.top_stocks,
        top_n_concepts=args.top_concepts
    )

    if args.date == 'latest':
        latest_date = signals['stock_anomaly']['date'].max() if not signals['stock_anomaly'].empty else 'latest'
        signals['snapshot_date'] = str(latest_date)[:10] if latest_date != 'latest' else 'latest'

    if args.db:
        save_signals_to_database(signals)
        print("数据库写入完成")

    output_path = export_signals_excel(signals, output_dir=args.output)
    print(f"Excel输出完成: {output_path}")