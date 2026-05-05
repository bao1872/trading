"""
market_structure_analysis
大盘结构性分析模块

Purpose:
    对 A 股市场整体及板块进行结构性分析，识别市场状态（趋势/震荡/分化）、
    板块轮动、风格切换、资金集中度等结构性特征。
    核心能力：个股因子+事件 → 全市场聚合 → 盘面状态分类 → 验证。

Modules:
    - factor_engine: 因子计算引擎（编排 DSA/BBMACD/Volume/PAVP/StopCluster）
    - event_detector: 事件检测器（12 核心事件 + 6 辅助状态）
    - batch_processor: 批量处理器（遍历股票池输出因子+事件表）
    - market_aggregator: 全市场聚合器（按日/行业/市值聚合事件统计 + 主导因子归因）
    - market_state_classifier: 市场状态分类器（4 主标签 + 2 副标签）
    - market_validator: 市场验证器（标签后收益验证 + 事件延续性验证）

Examples:
    批量处理 3 只股票:
        python market_structure_analysis/batch_processor.py --limit_stocks 3

    全市场聚合:
        python market_structure_analysis/market_aggregator.py --limit_stocks 10 --start 2024-06-01

    市场状态分类:
        python market_structure_analysis/market_state_classifier.py --limit_stocks 10 --start 2024-06-01

    标签验证:
        python market_structure_analysis/market_validator.py --start 2024-01-01 --end 2026-04-30
"""

__version__ = "0.3.0"
