"""
Purpose: 合并 signals_theme.py 和 signals_abnormal.py 同时运行
Inputs: dataset/, stock_concepts_cache.xlsx, theme_mapping.json
Outputs: 数据库表(7张)
Side Effects: 写库

Usage:
    # 写数据库
    python signals_combined.py --date 2026-04-02 --db

    # 小规模测试 (100只股票)
    python signals_combined.py --date 2026-04-02 --limit-stocks 100 --db

    # 回补信号（一次性加载数据，遍历日期计算）
    python signals_combined.py --backfill 2025-11-03 --db
    python signals_combined.py --backfill 2026-03-01 --end-date 2026-03-15 --db

Examples:
    python signals_combined.py --date 2026-04-02 --zscore 2.0 --top-themes 21 --top-stocks 200 --top-concepts 30 --db
"""
import pandas as pd
from datetime import datetime, timedelta

from signals_theme import generate_signals as theme_generate
from signals_theme import save_signals_to_database as theme_save
from signals_theme import print_signals as theme_print
from signals_abnormal import generate_rolling_signals as abnormal_generate
from signals_abnormal import save_signals_to_database as abnormal_save
from utils.concept_mapping import load_concept_data, build_stock_to_concepts
from datasource.k_data_loader import load_all_stock_data


def generate_combined_signals(
    period: str = 'daily',
    snapshot_date: str = None,
    zscore_threshold: float = 2.0,
    top_n_themes: int = 10,
    top_n_stocks: int = 20,
    top_n_concepts: int = 20,
    concept_xlsx_path: str = '../stock_concepts_cache.xlsx',
    limit_stocks: int = None
) -> dict:
    print("=" * 60)
    print("Step 1/4: 计算主题版信号 (向量化)")
    print("=" * 60)
    theme_signals = theme_generate(
        period=period,
        snapshot_date=snapshot_date,
        zscore_threshold=zscore_threshold,
        top_n_themes=top_n_themes,
        top_n_stocks=top_n_stocks,
        top_n_concepts=top_n_concepts,
        concept_xlsx_path=concept_xlsx_path,
        limit_stocks=limit_stocks
    )

    print("=" * 60)
    print("Step 2/4: 计算滚动窗口异动信号 (CV/Spearman)")
    print("=" * 60)
    abnormal_signals = abnormal_generate(
        freq=period,
        snapshot_date=snapshot_date,
        zscore_threshold=zscore_threshold,
        top_n_themes=top_n_themes,
        top_n_stocks=top_n_stocks,
        top_n_concepts=top_n_concepts,
        limit_stocks=limit_stocks
    )

    print("=" * 60)
    print("Step 3/4: 合并 CV/Spearman 到主题版 stock_anomaly")
    print("=" * 60)
    if not theme_signals['stock_anomaly'].empty and not abnormal_signals['stock_anomaly'].empty:
        cv_sp = abnormal_signals['stock_anomaly'][['code', 'volume_cv', 'volume_spearman']].copy()
        if 'volume_cv' not in theme_signals['stock_anomaly'].columns:
            theme_signals['stock_anomaly'] = theme_signals['stock_anomaly'].merge(
                cv_sp, on='code', how='left'
            )
    else:
        if 'volume_cv' not in theme_signals['stock_anomaly'].columns:
            theme_signals['stock_anomaly']['volume_cv'] = None
            theme_signals['stock_anomaly']['volume_spearman'] = None

    actual_date = theme_signals.get('snapshot_date') or abnormal_signals.get('snapshot_date') or snapshot_date

    print("=" * 60)
    print("Step 4/4: 加载概念映射 (供导出增强)")
    print("=" * 60)
    concept_df = load_concept_data(concept_xlsx_path)
    s2c = build_stock_to_concepts(concept_df)
    print(f"  概念映射加载完成: {len(s2c)} 只股票有概念映射")

    return {
        'theme_signals': theme_signals,
        'abnormal_signals': abnormal_signals,
        's2c': s2c,
        'snapshot_date': actual_date
    }


def save_combined_signals_to_database(signals: dict) -> None:
    print("写入 theme 版信号表 (theme_signals, concept_signals, stock_anomaly_signals, limit_up_signals)...")
    theme_signals_for_db = signals['theme_signals'].copy()
    theme_sa = theme_signals_for_db.get('stock_anomaly', pd.DataFrame())
    if not theme_sa.empty:
        drop_cols = [c for c in ['volume_cv', 'volume_spearman'] if c in theme_sa.columns]
        if drop_cols:
            theme_signals_for_db['stock_anomaly'] = theme_sa.drop(columns=drop_cols)
    theme_save(theme_signals_for_db)

    print("写入 abnormal 版信号表 (theme_signals_rolling, concept_signals_rolling, stock_anomaly_signals_rolling)...")
    abnormal_save(signals['abnormal_signals'])


def _compute_signals_for_date(
    period: str,
    snapshot_date: str,
    all_data: pd.DataFrame,
    name_map: dict,
    zscore_threshold: float,
    top_n_themes: int,
    top_n_stocks: int,
    top_n_concepts: int,
    concept_xlsx_path: str
) -> dict:
    """对单个截面日期计算信号，复用已加载数据，不重复加载"""
    preloaded = (all_data, name_map)

    print("=" * 60)
    print(f"计算主题版信号 [{snapshot_date}] (向量化)")
    print("=" * 60)
    theme_signals = theme_generate(
        period=period,
        snapshot_date=snapshot_date,
        zscore_threshold=zscore_threshold,
        top_n_themes=top_n_themes,
        top_n_stocks=top_n_stocks,
        top_n_concepts=top_n_concepts,
        concept_xlsx_path=concept_xlsx_path,
        preloaded_data=preloaded
    )

    print("=" * 60)
    print(f"计算滚动窗口异动信号 [{snapshot_date}] (CV/Spearman)")
    print("=" * 60)
    abnormal_signals = abnormal_generate(
        freq=period,
        snapshot_date=snapshot_date,
        zscore_threshold=zscore_threshold,
        top_n_themes=top_n_themes,
        top_n_stocks=top_n_stocks,
        top_n_concepts=top_n_concepts,
        preloaded_data=preloaded
    )

    print("=" * 60)
    print(f"合并 CV/Spearman 到主题版 stock_anomaly [{snapshot_date}]")
    print("=" * 60)
    if not theme_signals['stock_anomaly'].empty and not abnormal_signals['stock_anomaly'].empty:
        cv_sp = abnormal_signals['stock_anomaly'][['code', 'volume_cv', 'volume_spearman']].copy()
        if 'volume_cv' not in theme_signals['stock_anomaly'].columns:
            theme_signals['stock_anomaly'] = theme_signals['stock_anomaly'].merge(
                cv_sp, on='code', how='left'
            )
    else:
        if 'volume_cv' not in theme_signals['stock_anomaly'].columns:
            theme_signals['stock_anomaly']['volume_cv'] = None
            theme_signals['stock_anomaly']['volume_spearman'] = None

    concept_df = load_concept_data(concept_xlsx_path)
    s2c = build_stock_to_concepts(concept_df)

    return {
        'theme_signals': theme_signals,
        'abnormal_signals': abnormal_signals,
        's2c': s2c,
        'snapshot_date': snapshot_date
    }


def backfill_signals(
    start_date: str,
    end_date: str = None,
    period: str = 'daily',
    zscore_threshold: float = 2.0,
    top_n_themes: int = 10,
    top_n_stocks: int = 20,
    top_n_concepts: int = 20,
    db: bool = True,
    concept_xlsx_path: str = '../stock_concepts_cache.xlsx'
) -> None:
    """
    批量回补信号，一次加载数据，遍历日期计算

    Args:
        start_date: 回补开始日期
        end_date: 回补结束日期，默认为当天
        period: 周期
        zscore_threshold: Z-Score阈值
        top_n_themes: TOP主题数
        top_n_stocks: TOP股票数
        top_n_concepts: TOP概念数
        db: 是否写入数据库
        concept_xlsx_path: 概念映射文件路径
    """
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    lookback_start = (pd.Timestamp(start_date) - timedelta(days=60)).strftime('%Y-%m-%d')
    print("=" * 60)
    print(f"一次性加载数据: {lookback_start} 到 {end_date}")
    print("=" * 60)
    all_data, name_map = load_all_stock_data(
        freq=period,
        start_date=lookback_start,
        end_date=end_date
    )
    if all_data.empty:
        print("数据为空，回补终止")
        return
    print(f"数据加载完成: {len(all_data)} 行, {len(name_map)} 只股票")

    print("获取交易日列表...")
    trade_dates = all_data['bar_time'].dt.strftime('%Y-%m-%d').unique().tolist()
    trade_dates = sorted([d for d in trade_dates if start_date <= d <= end_date])

    if not trade_dates:
        print("无交易日，回补终止")
        return
    print(f"共 {len(trade_dates)} 个交易日待计算")

    try:
        from tqdm import tqdm
        iterator = tqdm(trade_dates, desc="回补进度")
    except ImportError:
        iterator = trade_dates

    for date in iterator:
        signals = _compute_signals_for_date(
            period=period,
            snapshot_date=date,
            all_data=all_data,
            name_map=name_map,
            zscore_threshold=zscore_threshold,
            top_n_themes=top_n_themes,
            top_n_stocks=top_n_stocks,
            top_n_concepts=top_n_concepts,
            concept_xlsx_path=concept_xlsx_path
        )

        if db:
            save_combined_signals_to_database(signals)

    print("=" * 60)
    print(f"回补完成: {trade_dates[0]} 到 {trade_dates[-1]}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='合并主题信号 + 滚动异动信号')
    parser.add_argument('--period', type=str, default='daily', choices=['daily', 'weekly', 'min60', 'min15'])
    parser.add_argument('--date', type=str, default=None, help='日期截面，如 2026-04-02')
    parser.add_argument('--backfill', type=str, default=None, help='回补开始日期，如 2026-03-01')
    parser.add_argument('--end-date', type=str, default=None, help='回补结束日期，默认为当天')
    parser.add_argument('--zscore', type=float, default=2.0, help='Z-Score阈值')
    parser.add_argument('--top-themes', type=int, default=10, help='TOP主题数')
    parser.add_argument('--top-stocks', type=int, default=20, help='TOP股票数')
    parser.add_argument('--top-concepts', type=int, default=20, help='TOP概念数')
    parser.add_argument('--db', action='store_true', help='输出到数据库')
    parser.add_argument('--print', action='store_true', help='打印主题版信号报告')
    parser.add_argument('--limit-stocks', type=int, default=None, help='限制股票数量（用于测试）')
    args = parser.parse_args()

    if args.backfill:
        backfill_signals(
            start_date=args.backfill,
            end_date=args.end_date,
            period=args.period,
            zscore_threshold=args.zscore,
            top_n_themes=args.top_themes,
            top_n_stocks=args.top_stocks,
            top_n_concepts=args.top_concepts,
            db=args.db
        )
    else:
        signals = generate_combined_signals(
            period=args.period,
            snapshot_date=args.date,
            zscore_threshold=args.zscore,
            top_n_themes=args.top_themes,
            top_n_stocks=args.top_stocks,
            top_n_concepts=args.top_concepts,
            limit_stocks=args.limit_stocks
        )

        if args.db:
            save_combined_signals_to_database(signals)
            print("数据库写入完成")

        if getattr(args, 'print', False):
            theme_print(signals['theme_signals'])
