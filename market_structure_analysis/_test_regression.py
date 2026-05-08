# market_structure_analysis/_test_regression.py
"""
Purpose: 最小测试闭环 — BUG / 设计问题 / 字段回归
Inputs: 无外部输入，全部使用合成数据
Outputs: 控制台输出测试结果（pass/fail）
How to Run: python market_structure_analysis/_test_regression.py
Examples:
    python market_structure_analysis/_test_regression.py
Side Effects: 无（不写 DB / 不写文件）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def test_empty_data_classify():
    """空数据场景: classify_market_state 不抛异常"""
    from market_structure_analysis.market_state_classifier import classify_market_state
    empty_daily = pd.DataFrame()
    result = classify_market_state(empty_daily, None)
    assert result.empty, f"空数据应返回空 DataFrame, 实际 len={len(result)}"
    print("[PASS] test_empty_data_classify")


def test_empty_data_report():
    """空数据场景: build_enhanced_daily_report 降级输出"""
    from market_structure_analysis.enhanced_daily_report import build_enhanced_daily_report
    empty_labeled = pd.DataFrame()
    report = build_enhanced_daily_report(empty_labeled)
    assert "A股盘面日报" in report, f"日报应包含标题, 实际: {report[:50]}"
    print("[PASS] test_empty_data_report")


def test_small_sample_size_style():
    """极小样本: size_style_label 不为误导性默认值"""
    from market_structure_analysis.market_state_classifier import (
        classify_market_state,
    )
    from market_structure_analysis._config import MIN_INDUSTRY_SAMPLE

    sample_daily = pd.DataFrame({
        "structure_score": [0.1, -0.05, 0.0],
        "momentum_score": [0.05, -0.02, 0.01],
        "volume_confirm_score": [0.0, 0.0, 0.0],
        "breakout_score": [0.0, 0.0, 0.0],
        "stop_flow_score": [0.0, 0.0, 0.0],
    }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))

    for col in [c + "_z20" for c in sample_daily.columns]:
        sample_daily[col] = 0.0

    result = classify_market_state(sample_daily, grouped_df=None)

    assert "size_style_label" in result.columns
    for label in result["size_style_label"]:
        assert label in ("大票占优", "小票占优", "未知"), f"无效标签: {label}"
    print("[PASS] test_small_sample_size_style")


def test_field_regression():
    """字段回归: labeled_df 包含所有必要列"""
    from market_structure_analysis.market_state_classifier import (
        classify_market_state,
    )

    n_rows = 10
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    sample_daily = pd.DataFrame({
        "structure_score": np.random.randn(n_rows) * 0.1,
        "momentum_score": np.random.randn(n_rows) * 0.1,
        "volume_confirm_score": np.random.randn(n_rows) * 0.05,
        "breakout_score": np.random.randn(n_rows) * 0.05,
        "stop_flow_score": np.random.randn(n_rows) * 0.05,
        "total_stocks": [200] * n_rows,
    }, index=dates)

    for col in [c + "_z20" for c in sample_daily.columns if c != "total_stocks"]:
        sample_daily[col] = sample_daily[col.replace("_z20", "")] * 2

    result = classify_market_state(sample_daily, grouped_df=None)

    required_cols = [
        "structure_score", "momentum_score",
        "main_label", "main_label_v2",
        "size_style_label", "breadth_label",
        "confidence",
    ]
    for col in required_cols:
        assert col in result.columns, f"缺少必要列: {col}"
    print("[PASS] test_field_regression")


def main():
    print("=" * 40)
    print("market_structure_analysis 最小测试")
    print("=" * 40)

    tests = [
        test_empty_data_classify,
        test_empty_data_report,
        test_small_sample_size_style,
        test_field_regression,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
            failed += 1

    print("-" * 40)
    print(f"结果: {passed}/{len(tests)} 通过, {failed}/{len(tests)} 失败")
    print("=" * 40)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
