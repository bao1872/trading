"""
Purpose: 主题聚合 - 根据主题映射汇总个股异动信号
Inputs: 放量Z-Score结果 + 概念映射 + 主题映射
Outputs: 主题强度排名、概念强度排名、主题内概念明细
"""
import numpy as np
import pandas as pd
import os
import json


def load_theme_mapping() -> dict:
    """加载主题映射配置"""
    json_path = os.path.join(os.path.dirname(__file__), '..', 'theme_mapping.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_concept_to_theme_map(mapping: dict) -> dict:
    """
    构建 底层概念 -> 主题 的映射
    """
    concept_to_theme = {}
    for theme_key, theme_info in mapping['themes'].items():
        theme_name = theme_info['name']
        for concept in theme_info.get('core_concepts', []):
            concept_to_theme[concept] = theme_name
        for concept in theme_info.get('related_concepts', []):
            if concept not in concept_to_theme:
                concept_to_theme[concept] = theme_name
    return concept_to_theme


def get_excluded_concepts(mapping: dict) -> set:
    """获取需要排除的概念集合"""
    return set(mapping.get('excluded_concepts', []))


def build_theme_to_concepts_map(mapping: dict) -> dict:
    """构建主题 -> 概念列表的映射"""
    theme_to_concepts = {}
    for theme_key, theme_info in mapping['themes'].items():
        theme_name = theme_info['name']
        concepts = theme_info.get('core_concepts', []) + theme_info.get('related_concepts', [])
        theme_to_concepts[theme_name] = concepts
    return theme_to_concepts


def aggregate_theme_scores(
    volume_zscore_df: pd.DataFrame,
    stock_to_concepts: dict,
    concept_to_stocks: dict,
    zscore_threshold: float = 2.0
) -> pd.DataFrame:
    """
    聚合主题强度（基于主题映射）

    Args:
        volume_zscore_df: 放量Z-Score结果
        stock_to_concepts: 股票 -> 概念列表
        concept_to_stocks: 概念 -> 股票列表（全量）
        zscore_threshold: Z-Score阈值

    Returns:
        DataFrame with theme ranking
    """
    mapping = load_theme_mapping()
    concept_to_theme = build_concept_to_theme_map(mapping)
    excluded = get_excluded_concepts(mapping)
    theme_to_concepts = build_theme_to_concepts_map(mapping)

    anomalous = volume_zscore_df[volume_zscore_df['zscore'] > zscore_threshold].copy()

    theme_scores = {}
    theme_concept_detail = {}

    for _, row in anomalous.iterrows():
        code = row['code']
        zscore = row['zscore']
        concepts = stock_to_concepts.get(code, [])

        stock_themes = set()
        stock_concepts_in_theme = {}
        for concept in concepts:
            if concept in excluded:
                continue
            theme = concept_to_theme.get(concept)
            if theme:
                stock_themes.add(theme)
                if theme not in stock_concepts_in_theme:
                    stock_concepts_in_theme[theme] = []
                stock_concepts_in_theme[theme].append(concept)

        for theme in stock_themes:
            if theme not in theme_scores:
                theme_scores[theme] = {
                    'total_zscore': 0.0,
                    'stocks': [],
                    'zscore_list': [],
                    'anomalous_concepts': set(),
                    'all_concepts': set(theme_to_concepts.get(theme, []))
                }
            theme_scores[theme]['total_zscore'] += zscore
            theme_scores[theme]['stocks'].append(code)
            theme_scores[theme]['zscore_list'].append(zscore)
            theme_scores[theme]['anomalous_concepts'].update(stock_concepts_in_theme.get(theme, []))

    results = []
    for theme, info in theme_scores.items():
        stock_count = len(info['stocks'])
        total_zscore = info['total_zscore']
        strength = total_zscore / np.sqrt(stock_count) if stock_count > 0 else 0

        all_concepts = info['all_concepts']
        anomalous_concepts = info['anomalous_concepts']
        concept_coverage = len(anomalous_concepts) / len(all_concepts) if all_concepts else 0

        top_indices = np.argsort(info['zscore_list'])[-3:][::-1]
        top_stocks = [info['stocks'][i] for i in top_indices]

        results.append({
            'theme': theme,
            'total_zscore': total_zscore,
            'stock_count': stock_count,
            'strength': strength,
            'anomalous_concept_count': len(anomalous_concepts),
            'total_concept_count': len(all_concepts),
            'concept_coverage': concept_coverage,
            'top_stocks': top_stocks
        })

    if not results:
        return pd.DataFrame(columns=['theme', 'total_zscore', 'stock_count', 'strength',
                                     'anomalous_concept_count', 'total_concept_count', 'concept_coverage', 'top_stocks'])

    return pd.DataFrame(results).sort_values('strength', ascending=False).reset_index(drop=True)


def aggregate_concept_scores(
    volume_zscore_df: pd.DataFrame,
    stock_to_concepts: dict,
    concept_to_stocks: dict,
    zscore_threshold: float = 2.0
) -> pd.DataFrame:
    """
    聚合概念强度（归一化，按成分股数量）

    Args:
        volume_zscore_df: 放量Z-Score结果
        stock_to_concepts: 股票 -> 概念列表
        concept_to_stocks: 概念 -> 股票列表（全量）
        zscore_threshold: Z-Score阈值

    Returns:
        DataFrame with concept ranking
    """
    mapping = load_theme_mapping()
    concept_to_theme = build_concept_to_theme_map(mapping)
    excluded = get_excluded_concepts(mapping)

    anomalous = volume_zscore_df[volume_zscore_df['zscore'] > zscore_threshold].copy()

    concept_scores = {}

    for _, row in anomalous.iterrows():
        code = row['code']
        zscore = row['zscore']
        concepts = stock_to_concepts.get(code, [])

        for concept in concepts:
            if concept in excluded:
                continue
            if concept not in concept_scores:
                concept_scores[concept] = {
                    'total_zscore': 0.0,
                    'stocks': [],
                    'zscore_list': [],
                    'total_stocks': len(concept_to_stocks.get(concept, []))
                }
            concept_scores[concept]['total_zscore'] += zscore
            concept_scores[concept]['stocks'].append(code)
            concept_scores[concept]['zscore_list'].append(zscore)

    results = []
    for concept, info in concept_scores.items():
        stock_count = len(info['stocks'])
        total_zscore = info['total_zscore']
        total_stocks = info['total_stocks']

        avg_zscore = total_zscore / stock_count if stock_count > 0 else 0
        normalized_strength = total_zscore / np.sqrt(stock_count) if stock_count > 0 else 0
        intensity = stock_count / total_stocks if total_stocks > 0 else 0

        theme = concept_to_theme.get(concept, '其他')

        top_indices = np.argsort(info['zscore_list'])[-3:][::-1]
        top_stocks = [info['stocks'][i] for i in top_indices]

        results.append({
            'concept': concept,
            'theme': theme,
            'total_zscore': total_zscore,
            'anomalous_stock_count': stock_count,
            'total_stock_count': total_stocks,
            'avg_zscore': avg_zscore,
            'normalized_strength': normalized_strength,
            'intensity': intensity,
            'top_stocks': top_stocks
        })

    if not results:
        return pd.DataFrame(columns=['concept', 'theme', 'total_zscore', 'anomalous_stock_count',
                                     'total_stock_count', 'avg_zscore', 'normalized_strength', 'intensity', 'top_stocks'])

    return pd.DataFrame(results).sort_values('normalized_strength', ascending=False).reset_index(drop=True)


def aggregate_theme_with_concept_detail(
    volume_zscore_df: pd.DataFrame,
    stock_to_concepts: dict,
    concept_to_stocks: dict,
    zscore_threshold: float = 2.0
) -> tuple:
    """
    主题聚合 + 概念明细 + 显著性检测

    Returns:
        (theme_df, concept_df, theme_concept_detail_df)
    """
    mapping = load_theme_mapping()
    concept_to_theme = build_concept_to_theme_map(mapping)
    excluded = get_excluded_concepts(mapping)
    theme_to_concepts = build_theme_to_concepts_map(mapping)

    anomalous = volume_zscore_df[volume_zscore_df['zscore'] > zscore_threshold].copy()

    theme_data = {}
    concept_data = {}
    theme_concept_detail = {}

    for _, row in anomalous.iterrows():
        code = row['code']
        zscore = row['zscore']
        concepts = stock_to_concepts.get(code, [])

        stock_themes = set()
        for concept in concepts:
            if concept in excluded:
                continue
            theme = concept_to_theme.get(concept)
            if theme:
                stock_themes.add(theme)

                if concept not in concept_data:
                    concept_data[concept] = {
                        'total_zscore': 0.0,
                        'stocks': [],
                        'zscore_list': [],
                        'total_stocks': len(concept_to_stocks.get(concept, []))
                    }
                concept_data[concept]['total_zscore'] += zscore
                concept_data[concept]['stocks'].append(code)
                concept_data[concept]['zscore_list'].append(zscore)

        for theme in stock_themes:
            if theme not in theme_data:
                theme_data[theme] = {
                    'total_zscore': 0.0,
                    'stocks': [],
                    'zscore_list': [],
                    'all_concepts': set(theme_to_concepts.get(theme, [])),
                    'anomalous_concepts': set()
                }
            theme_data[theme]['total_zscore'] += zscore
            theme_data[theme]['stocks'].append(code)
            theme_data[theme]['zscore_list'].append(zscore)
            for concept in concepts:
                if concept in excluded:
                    continue
                if concept_to_theme.get(concept) == theme:
                    theme_data[theme]['anomalous_concepts'].add(concept)

    theme_results = []
    for theme, info in theme_data.items():
        stock_count = len(info['stocks'])
        total_zscore = info['total_zscore']
        avg_zscore = total_zscore / stock_count if stock_count > 0 else 0
        strength = total_zscore / np.sqrt(stock_count) if stock_count > 0 else 0
        all_concepts = info['all_concepts']
        anomalous_concepts = info['anomalous_concepts']
        concept_coverage = len(anomalous_concepts) / len(all_concepts) if all_concepts else 0

        top_indices = np.argsort(info['zscore_list'])[-3:][::-1]
        top_stocks = [info['stocks'][i] for i in top_indices]

        theme_results.append({
            'theme': theme,
            'total_zscore': total_zscore,
            'avg_zscore': avg_zscore,
            'stock_count': stock_count,
            'strength': strength,
            'anomalous_concept_count': len(anomalous_concepts),
            'total_concept_count': len(all_concepts),
            'concept_coverage': concept_coverage,
            'top_stocks': top_stocks
        })

    concept_results = []
    for concept, info in concept_data.items():
        stock_count = len(info['stocks'])
        total_zscore = info['total_zscore']
        total_stocks = info['total_stocks']
        avg_zscore = total_zscore / stock_count if stock_count > 0 else 0
        normalized_strength = total_zscore / np.sqrt(stock_count) if stock_count > 0 else 0
        intensity = stock_count / total_stocks if total_stocks > 0 else 0
        theme = concept_to_theme.get(concept, '其他')

        top_indices = np.argsort(info['zscore_list'])[-3:][::-1]
        top_stocks = [info['stocks'][i] for i in top_indices]

        concept_results.append({
            'concept': concept,
            'theme': theme,
            'total_zscore': total_zscore,
            'avg_zscore': avg_zscore,
            'anomalous_stock_count': stock_count,
            'total_stock_count': total_stocks,
            'normalized_strength': normalized_strength,
            'intensity': intensity,
            'top_stocks': top_stocks
        })

    theme_df = pd.DataFrame(theme_results).sort_values('stock_count', ascending=False).reset_index(drop=True) if theme_results else pd.DataFrame()
    concept_df = pd.DataFrame(concept_results).sort_values('anomalous_stock_count', ascending=False).reset_index(drop=True) if concept_results else pd.DataFrame()

    detail_results = []
    for theme, info in theme_data.items():
        all_concepts = info['all_concepts']
        anomalous_concepts = info['anomalous_concepts']
        theme_concept_data = concept_data

        for concept in all_concepts:
            if concept in anomalous_concepts:
                c_info = concept_data.get(concept, {})
                c_stock_count = len(c_info.get('stocks', []))
                c_total_zscore = c_info.get('total_zscore', 0)
                c_total_stocks = c_info.get('total_stocks', 0)
                c_intensity = c_stock_count / c_total_stocks if c_total_stocks > 0 else 0
                c_normalized = c_total_zscore / np.sqrt(c_stock_count) if c_stock_count > 0 else 0

                detail_results.append({
                    'theme': theme,
                    'concept': concept,
                    'is_anomalous': True,
                    'anomalous_stock_count': c_stock_count,
                    'total_stock_count': c_total_stocks,
                    'intensity': c_intensity,
                    'total_zscore': c_total_zscore,
                    'normalized_strength': c_normalized
                })

    detail_df = pd.DataFrame(detail_results) if detail_results else pd.DataFrame()

    return theme_df, concept_df, detail_df


def get_stock_themes(stock_concepts: list, concept_to_theme: dict, excluded: set) -> list:
    """获取股票所属的主题列表"""
    themes = []
    for concept in stock_concepts:
        if concept in excluded:
            continue
        theme = concept_to_theme.get(concept)
        if theme and theme not in themes:
            themes.append(theme)
    return themes


if __name__ == '__main__':
    from volume_anomaly import calculate_volume_zscore, load_data_for_period
    from concept_mapping import load_concept_data, build_stock_to_concepts, build_concept_to_stocks

    daily_data = load_data_for_period('daily')
    volume_df = calculate_volume_zscore(daily_data, window=42)

    concept_df = load_concept_data('../../stock_concepts_cache.xlsx')
    s2c = build_stock_to_concepts(concept_df)
    c2s = build_concept_to_stocks(concept_df)

    theme_df, concept_df, detail_df = aggregate_theme_with_concept_detail(volume_df, s2c, c2s)

    print("=== Theme Ranking ===")
    print(theme_df.head(10).to_string())

    print("\n=== Concept Ranking (Normalized) ===")
    print(concept_df.head(10).to_string())
