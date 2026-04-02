"""
Purpose: 主题映射 - 加载和使用theme_mapping.json
Inputs: theme_mapping.json
Outputs: 股票->主题映射、主题->概念映射
"""
import json
import os


def load_theme_mapping(json_path: str = None) -> dict:
    """加载主题映射配置"""
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), 'theme_mapping.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_concept_to_theme_map(mapping: dict) -> dict:
    """
    构建 底层概念 -> 主题 的映射
    支持core_concepts和related_concepts
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


def get_stock_themes(stock_concepts: list, concept_to_theme: dict, excluded: set) -> list:
    """
    获取股票所属的主题列表

    Args:
        stock_concepts: 股票的底层概念列表
        concept_to_theme: 概念->主题映射
        excluded: 需要排除的概念集合

    Returns:
        主题列表（去重）
    """
    themes = []
    for concept in stock_concepts:
        if concept in excluded:
            continue
        theme = concept_to_theme.get(concept)
        if theme and theme not in themes:
            themes.append(theme)
    return themes


def get_theme_concepts(theme_key: str, mapping: dict) -> list:
    """获取指定主题的所有底层概念"""
    theme_info = mapping['themes'].get(theme_key, {})
    return theme_info.get('core_concepts', []) + theme_info.get('related_concepts', [])


if __name__ == '__main__':
    mapping = load_theme_mapping()
    concept_to_theme = build_concept_to_theme_map(mapping)
    excluded = set(mapping.get('excluded_concepts', []))

    print(f"主题数: {len(mapping['themes'])}")
    print(f"已映射概念数: {len(concept_to_theme)}")
    print(f"已排除概念数: {len(excluded)}")

    print("\n示例 - AI主题的概念:")
    ai_concepts = get_theme_concepts('ai', mapping)
    for c in ai_concepts[:10]:
        print(f"  {c} -> {concept_to_theme.get(c)}")
