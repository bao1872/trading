"""
Purpose: 合并 signals_theme.py 和 signals_abnormal.py 同时运行
Inputs: dataset/, stock_concepts_cache.xlsx, theme_mapping.json
Outputs: theme_{date}.xlsx, signal_{date}.xlsx, 数据库表(7张)
Side Effects: 写库、写Excel

Usage:
    # 同时生成两个Excel
    python signals_combined.py --date 2026-03-20

    # 同时写数据库 + 生成Excel
    python signals_combined.py --date 2026-03-20 --db --output ../review

    # 使用最新日期
    python signals_combined.py --date latest --db

Examples:
    python signals_combined.py --date 2026-03-20 --zscore 2.0 --top-themes 21 --top-stocks 200 --top-concepts 30 --db
"""
import os
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
    concept_xlsx_path: str = '../stock_concepts_cache.xlsx'
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
        concept_xlsx_path=concept_xlsx_path
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
        top_n_concepts=top_n_concepts
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


def _export_theme_excel_with_concepts(
    theme_signals: dict,
    s2c: dict,
    output_dir: str
) -> str:
    from datasource.k_data_loader import build_name_map

    period = theme_signals['period']
    snapshot_date = theme_signals['snapshot_date']
    output_path = os.path.join(output_dir, f'theme_{snapshot_date}.xlsx')

    code_to_name = build_name_map()

    column_rename = {
        'theme': '主题', 'total_zscore': '总Z分数', 'stock_count': '股票数量',
        'strength': '强度', 'anomalous_concept_count': '异动概念数',
        'total_concept_count': '总概念数', 'concept_coverage': '概念覆盖率',
        'top_stocks': 'TOP股票', 'concept': '概念',
        'anomalous_stock_count': '异动股数', 'total_stock_count': '总股数',
        'avg_zscore': '平均Z分数', 'normalized_strength': '标准化强度',
        'intensity': '强度', 'zscore': 'Z分数', 'volume': '成交量',
        'rolling_mean': '滚动均值', 'date': '日期',
        'themes': '主题列表', 'themes_str': '主题',
        'volume_cv': '成交量均匀度(CV)', 'volume_spearman': '放量顺序(ρ)',
        'concepts_str': '概念列表',
    }

    def rename_columns(df):
        return df.rename(columns=column_rename)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        theme_df = theme_signals['theme_ranking']
        if not theme_df.empty:
            theme_export = theme_df.head(21).copy()
            if 'top_stocks' in theme_export.columns:
                def format_stocks(codes):
                    if isinstance(codes, list):
                        return ', '.join([f"{code_to_name.get(c, c)}({c})" for c in codes])
                    return codes
                theme_export['top_stocks'] = theme_export['top_stocks'].apply(format_stocks)
            rename_columns(theme_export).to_excel(writer, sheet_name='主题排名', index=False)

        concept_df = theme_signals['concept_ranking']
        if not concept_df.empty:
            concept_export = concept_df.head(20).copy()
            if 'top_stocks' in concept_export.columns:
                def format_stocks(codes):
                    if isinstance(codes, list):
                        return ', '.join([f"{code_to_name.get(c, c)}({c})" for c in codes])
                    return codes
                concept_export['top_stocks'] = concept_export['top_stocks'].apply(format_stocks)
            rename_columns(concept_export).to_excel(writer, sheet_name='概念排名', index=False)

        stock_df = theme_signals['stock_anomaly']
        if not stock_df.empty:
            stock_export = stock_df[stock_df['zscore'] > 2.5].copy()
            if not stock_export.empty:
                if 'name' in stock_export.columns and 'code' in stock_export.columns:
                    stock_export['股票'] = stock_export['name'] + '(' + stock_export['code'] + ')'
                    cols = ['股票'] + [c for c in stock_export.columns if c not in ['name', 'code', '股票']]
                    stock_export = stock_export[cols]
                rename_columns(stock_export).to_excel(writer, sheet_name='个股异动', index=False)

        limit_up_df = theme_signals['theme_limit_up']
        if not limit_up_df.empty:
            rows = []
            for _, row in limit_up_df.iterrows():
                theme = row['theme']
                streak_groups = row.get('streak_groups', {})
                ordered_keys = sorted(streak_groups.keys(), key=_streak_sort_key)
                for key in ordered_keys:
                    if key in streak_groups and streak_groups[key]:
                        stocks = streak_groups[key]
                        for s in stocks:
                            code = s['code']
                            concepts = s2c.get(code, [])
                            concepts_str = ','.join(concepts) if concepts else ''
                            names = f"{s['name']}({s['code']})"
                            rows.append({'主题': theme, '几天几板': key, '股票名称': names, '概念': concepts_str})
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name='涨停金字塔', index=False)

        limit_down_df = theme_signals['theme_limit_down']
        if not limit_down_df.empty:
            rows = []
            for _, row in limit_down_df.iterrows():
                theme = row['theme']
                streak_groups = row.get('streak_groups', {})
                ordered_keys = sorted(streak_groups.keys(), key=_streak_sort_key)
                for key in ordered_keys:
                    if key in streak_groups and streak_groups[key]:
                        stocks = streak_groups[key]
                        for s in stocks:
                            code = s['code']
                            concepts = s2c.get(code, [])
                            concepts_str = ','.join(concepts) if concepts else ''
                            names = f"{s['name']}({s['code']})"
                            rows.append({'主题': theme, '几天几板': key, '股票名称': names, '概念': concepts_str})
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name='跌停金字塔', index=False)

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


def _streak_sort_key(key):
    import re
    m = re.match(r'(\d+)天(\d+)板', key)
    if m:
        return (-int(m.group(2)), int(m.group(1)))
    return (0, 0)


def export_combined_signals_excel(signals: dict, output_dir: str = '../review') -> tuple:
    print("=" * 60)
    print("导出 theme 版 Excel (含概念增强)")
    print("=" * 60)
    theme_path = _export_theme_excel_with_concepts(
        signals['theme_signals'], signals['s2c'], output_dir
    )
    print(f"  主题版Excel: {theme_path}")

    print("=" * 60)
    print("导出 abnormal 版 Excel")
    print("=" * 60)
    from signals_abnormal import export_signals_excel as abnormal_export
    abnormal_path = abnormal_export(signals['abnormal_signals'], output_dir)
    print(f"  异动版Excel: {abnormal_path}")

    return theme_path, abnormal_path


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='合并主题信号 + 滚动异动信号')
    parser.add_argument('--period', type=str, default='daily', choices=['daily', 'weekly', 'min60', 'min15'])
    parser.add_argument('--date', type=str, default=None, help='日期截面，如 2026-03-20')
    parser.add_argument('--zscore', type=float, default=2.0, help='Z-Score阈值')
    parser.add_argument('--top-themes', type=int, default=10, help='TOP主题数')
    parser.add_argument('--top-stocks', type=int, default=20, help='TOP股票数')
    parser.add_argument('--top-concepts', type=int, default=20, help='TOP概念数')
    parser.add_argument('--db', action='store_true', help='输出到数据库')
    parser.add_argument('--output', type=str, default='../review', help='输出目录')
    parser.add_argument('--print', action='store_true', help='打印主题版信号报告')
    args = parser.parse_args()

    abs_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(abs_dir, args.output) if not os.path.isabs(args.output) else args.output

    signals = generate_combined_signals(
        period=args.period,
        snapshot_date=args.date,
        zscore_threshold=args.zscore,
        top_n_themes=args.top_themes,
        top_n_stocks=args.top_stocks,
        top_n_concepts=args.top_concepts
    )

    if args.db:
        save_combined_signals_to_database(signals)
        print("数据库写入完成")

    export_combined_signals_excel(signals, output_dir=output_dir)

    if getattr(args, 'print', False):
        theme_print(signals['theme_signals'])
