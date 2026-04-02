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

Examples:
    python signals_combined.py --date 2026-04-02 --zscore 2.0 --top-themes 21 --top-stocks 200 --top-concepts 30 --db
"""
import pandas as pd

from signals_theme import generate_signals as theme_generate
from signals_theme import save_signals_to_database as theme_save
from signals_theme import print_signals as theme_print
from signals_abnormal import generate_rolling_signals as abnormal_generate
from signals_abnormal import save_signals_to_database as abnormal_save
from utils.concept_mapping import load_concept_data, build_stock_to_concepts


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='合并主题信号 + 滚动异动信号')
    parser.add_argument('--period', type=str, default='daily', choices=['daily', 'weekly', 'min60', 'min15'])
    parser.add_argument('--date', type=str, default=None, help='日期截面，如 2026-04-02')
    parser.add_argument('--zscore', type=float, default=2.0, help='Z-Score阈值')
    parser.add_argument('--top-themes', type=int, default=10, help='TOP主题数')
    parser.add_argument('--top-stocks', type=int, default=20, help='TOP股票数')
    parser.add_argument('--top-concepts', type=int, default=20, help='TOP概念数')
    parser.add_argument('--db', action='store_true', help='输出到数据库')
    parser.add_argument('--print', action='store_true', help='打印主题版信号报告')
    parser.add_argument('--limit-stocks', type=int, default=None, help='限制股票数量（用于测试）')
    args = parser.parse_args()

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
