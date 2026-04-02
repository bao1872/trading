"""
Purpose: 概念映射 - 股票代码与概念的双向映射
Inputs: 数据库 stock_concepts_cache 表
Outputs: stock_to_concepts, concept_to_stocks 字典
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_concept_data(xlsx_path: str = None) -> pd.DataFrame:
    """加载概念数据（从数据库）"""
    from datasource.database import get_session, query_df
    with get_session() as session:
        df = query_df(session, table_name="stock_pools")
    return df


def build_stock_to_concepts(df: pd.DataFrame) -> dict:
    """
    构建 股票代码 -> 概念列表 的映射
    concepts列以分号分隔
    同时返回纯数字代码和带后缀代码的映射
    """
    stock_to_concepts = {}
    for _, row in df.iterrows():
        code_with_suffix = row['ts_code']
        code_without_suffix = code_with_suffix.split('.')[0] if '.' in code_with_suffix else code_with_suffix
        concepts_str = row.get('concepts', '')
        if pd.isna(concepts_str) or not concepts_str:
            stock_to_concepts[code_with_suffix] = []
            stock_to_concepts[code_without_suffix] = []
        else:
            concepts_list = [c.strip() for c in str(concepts_str).split(';') if c.strip()]
            stock_to_concepts[code_with_suffix] = concepts_list
            stock_to_concepts[code_without_suffix] = concepts_list
    return stock_to_concepts


def build_concept_to_stocks(df: pd.DataFrame) -> dict:
    """
    构建 概念 -> 股票列表 的映射（反推）
    """
    concept_to_stocks = {}
    for _, row in df.iterrows():
        code_with_suffix = row['ts_code']
        code_without_suffix = code_with_suffix.split('.')[0] if '.' in code_with_suffix else code_with_suffix
        concepts_str = row.get('concepts', '')
        if pd.isna(concepts_str) or not concepts_str:
            continue
        for concept in str(concepts_str).split(';'):
            concept = concept.strip()
            if not concept:
                continue
            if concept not in concept_to_stocks:
                concept_to_stocks[concept] = []
            concept_to_stocks[concept].append(code_without_suffix)
    return concept_to_stocks


if __name__ == '__main__':
    df = load_concept_data()
    s2c = build_stock_to_concepts(df)
    c2s = build_concept_to_stocks(df)
    print(f'股票数: {len(s2c)}, 概念数: {len(c2s)}')
    print('示例 - 华为概念成分股:', c2s.get('华为概念', [])[:5])
