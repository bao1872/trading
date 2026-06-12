# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_structure.py - SR结构类因子

Purpose: SR结构类因子（支撑/压力位、翻转、触碰、新鲜度等）的批量计算与注册。

Public API:
    compute_sr_structure_factors(df, pivot_len=10) -> DataFrame

Registered Factors:
    - support_ref: 当前有效支撑位(=active_support_ref)
    - resistance_ref: 最近压力位
    - pivot_support_ref: pivot low 支撑位
    - flipped_support_ref: 压力转支撑位
    - active_support_ref: 当前有效支撑位
    - is_support_flipped: 当前支撑是否为压力转支撑
    - prev_support_ref: 上一个支撑位
    - prev_resistance_ref: 上一个压力位
    - support_age_bars: 支撑确认后经过K线数
    - resistance_age_bars: 压力确认后经过K线数
    - flipped_support_age_bars: 压力转支撑形成后经过K线数
    - support_is_higher_low: 支撑是否抬高
    - resistance_is_higher_high: 压力是否抬高
    - support_touch_count_20: 20期触碰支撑次数
    - support_touch_count_60: 60期触碰支撑次数
    - resistance_touch_count_20: 20期触碰压力次数
    - resistance_touch_count_60: 60期触碰压力次数
    - support_is_fresh: 支撑是否新鲜
    - resistance_is_fresh: 压力是否新鲜
    - support_is_overused: 支撑是否过度使用
    - resistance_is_overused: 压力是否过度使用
    - support_gap_pct: flipped_support与pivot_support间距百分比

Inputs:
    df: 含 open/high/low/close/volume 列的 DataFrame

Outputs:
    DataFrame 含上述因子列

How to Run:
    python -m factor_lib.categories.sr_structure          # 自测
    python -m factor_lib.categories.sr_structure --help

Examples:
    from factor_lib.categories.sr_structure import compute_sr_structure_factors
    result = compute_sr_structure_factors(df_kline)

    from factor_lib import compute_panel_v2
    factors = compute_panel_v2(df_kline, categories=["SR结构类"])

Side Effects: 无（纯计算，不写库/改表/写文件）
"""
from factor_lib.registry import register_factor
import pandas as pd

_SR_FACTOR_COLUMNS = [
    "support_ref",
    "resistance_ref",
    "pivot_support_ref",
    "flipped_support_ref",
    "active_support_ref",
    "is_support_flipped",
    "prev_support_ref",
    "prev_resistance_ref",
    "support_age_bars",
    "resistance_age_bars",
    "flipped_support_age_bars",
    "support_is_higher_low",
    "resistance_is_higher_high",
    "support_touch_count_20",
    "support_touch_count_60",
    "resistance_touch_count_20",
    "resistance_touch_count_60",
    "support_is_fresh",
    "resistance_is_fresh",
    "support_is_overused",
    "resistance_is_overused",
    "support_gap_pct",
    "support_cluster_count",
    "support_cluster_score",
    "support_cluster_density",
    "support_zone_low",
    "support_zone_high",
    "support_cluster_is_strong",
    "resistance_cluster_count",
    "resistance_cluster_score",
    "resistance_zone_low",
    "resistance_zone_high",
    "resistance_cluster_is_strong",
    "support_confluence_score",
    "resistance_confluence_score",
    "support_confluence_is_strong",
    "resistance_confluence_is_strong",
]

_COLUMN_MAP = {
    "support_ref": "active_support_ref",
}


def compute_sr_structure_factors(df: pd.DataFrame, pivot_len: int = 10) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    cfg = LabConfig(pivot_len=pivot_len)
    sr_result = compute_sr_factor_lab(df, cfg)
    result = pd.DataFrame(index=df.index)
    for col in _SR_FACTOR_COLUMNS:
        src_col = _COLUMN_MAP.get(col, col)
        if src_col in sr_result.columns:
            result[col] = sr_result[src_col]
    return result


def _compute_support_ref(df):
    return compute_sr_structure_factors(df)["support_ref"]


def _compute_resistance_ref(df):
    return compute_sr_structure_factors(df)["resistance_ref"]


def _compute_pivot_support_ref(df):
    return compute_sr_structure_factors(df)["pivot_support_ref"]


def _compute_flipped_support_ref(df):
    return compute_sr_structure_factors(df)["flipped_support_ref"]


def _compute_active_support_ref(df):
    return compute_sr_structure_factors(df)["active_support_ref"]


def _compute_is_support_flipped(df):
    return compute_sr_structure_factors(df)["is_support_flipped"]


def _compute_prev_support_ref(df):
    return compute_sr_structure_factors(df)["prev_support_ref"]


def _compute_prev_resistance_ref(df):
    return compute_sr_structure_factors(df)["prev_resistance_ref"]


def _compute_support_age_bars(df):
    return compute_sr_structure_factors(df)["support_age_bars"]


def _compute_resistance_age_bars(df):
    return compute_sr_structure_factors(df)["resistance_age_bars"]


def _compute_flipped_support_age_bars(df):
    return compute_sr_structure_factors(df)["flipped_support_age_bars"]


def _compute_support_is_higher_low(df):
    return compute_sr_structure_factors(df)["support_is_higher_low"]


def _compute_resistance_is_higher_high(df):
    return compute_sr_structure_factors(df)["resistance_is_higher_high"]


def _compute_support_touch_count_20(df):
    return compute_sr_structure_factors(df)["support_touch_count_20"]


def _compute_support_touch_count_60(df):
    return compute_sr_structure_factors(df)["support_touch_count_60"]


def _compute_resistance_touch_count_20(df):
    return compute_sr_structure_factors(df)["resistance_touch_count_20"]


def _compute_resistance_touch_count_60(df):
    return compute_sr_structure_factors(df)["resistance_touch_count_60"]


def _compute_support_is_fresh(df):
    return compute_sr_structure_factors(df)["support_is_fresh"]


def _compute_resistance_is_fresh(df):
    return compute_sr_structure_factors(df)["resistance_is_fresh"]


def _compute_support_is_overused(df):
    return compute_sr_structure_factors(df)["support_is_overused"]


def _compute_resistance_is_overused(df):
    return compute_sr_structure_factors(df)["resistance_is_overused"]


def _compute_support_gap_pct(df):
    return compute_sr_structure_factors(df)["support_gap_pct"]


def _compute_support_cluster_count(df):
    return compute_sr_structure_factors(df)["support_cluster_count"]


def _compute_support_cluster_score(df):
    return compute_sr_structure_factors(df)["support_cluster_score"]


def _compute_support_cluster_density(df):
    return compute_sr_structure_factors(df)["support_cluster_density"]


def _compute_support_zone_low(df):
    return compute_sr_structure_factors(df)["support_zone_low"]


def _compute_support_zone_high(df):
    return compute_sr_structure_factors(df)["support_zone_high"]


def _compute_support_cluster_is_strong(df):
    return compute_sr_structure_factors(df)["support_cluster_is_strong"]


def _compute_resistance_cluster_count(df):
    return compute_sr_structure_factors(df)["resistance_cluster_count"]


def _compute_resistance_cluster_score(df):
    return compute_sr_structure_factors(df)["resistance_cluster_score"]


def _compute_resistance_zone_low(df):
    return compute_sr_structure_factors(df)["resistance_zone_low"]


def _compute_resistance_zone_high(df):
    return compute_sr_structure_factors(df)["resistance_zone_high"]


def _compute_resistance_cluster_is_strong(df):
    return compute_sr_structure_factors(df)["resistance_cluster_is_strong"]


def _compute_support_confluence_score(df):
    return compute_sr_structure_factors(df)["support_confluence_score"]


def _compute_resistance_confluence_score(df):
    return compute_sr_structure_factors(df)["resistance_confluence_score"]


def _compute_support_confluence_is_strong(df):
    return compute_sr_structure_factors(df)["support_confluence_is_strong"]


def _compute_resistance_confluence_is_strong(df):
    return compute_sr_structure_factors(df)["resistance_confluence_is_strong"]


register_factor(
    name="support_ref",
    category="SR结构类",
    compute_func=_compute_support_ref,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="当前有效支撑位(=active_support_ref)",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="resistance_ref",
    category="SR结构类",
    compute_func=_compute_resistance_ref,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="最近压力位",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="pivot_support_ref",
    category="SR结构类",
    compute_func=_compute_pivot_support_ref,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="pivot low 支撑位",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="flipped_support_ref",
    category="SR结构类",
    compute_func=_compute_flipped_support_ref,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力转支撑位",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="active_support_ref",
    category="SR结构类",
    compute_func=_compute_active_support_ref,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="当前有效支撑位",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="is_support_flipped",
    category="SR结构类",
    compute_func=_compute_is_support_flipped,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="当前支撑是否为压力转支撑",
    direction="positive",
    is_core=True,
)

register_factor(
    name="prev_support_ref",
    category="SR结构类",
    compute_func=_compute_prev_support_ref,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="上一个支撑位",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="prev_resistance_ref",
    category="SR结构类",
    compute_func=_compute_prev_resistance_ref,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="上一个压力位",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="support_age_bars",
    category="SR结构类",
    compute_func=_compute_support_age_bars,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑确认后经过K线数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="resistance_age_bars",
    category="SR结构类",
    compute_func=_compute_resistance_age_bars,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力确认后经过K线数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="flipped_support_age_bars",
    category="SR结构类",
    compute_func=_compute_flipped_support_age_bars,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力转支撑形成后经过K线数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="support_is_higher_low",
    category="SR结构类",
    compute_func=_compute_support_is_higher_low,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑是否抬高",
    direction="positive",
    is_core=True,
)

register_factor(
    name="resistance_is_higher_high",
    category="SR结构类",
    compute_func=_compute_resistance_is_higher_high,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力是否抬高",
    direction="positive",
    is_core=True,
)

register_factor(
    name="support_touch_count_20",
    category="SR结构类",
    compute_func=_compute_support_touch_count_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期触碰支撑次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="support_touch_count_60",
    category="SR结构类",
    compute_func=_compute_support_touch_count_60,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="60期触碰支撑次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="resistance_touch_count_20",
    category="SR结构类",
    compute_func=_compute_resistance_touch_count_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期触碰压力次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="resistance_touch_count_60",
    category="SR结构类",
    compute_func=_compute_resistance_touch_count_60,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="60期触碰压力次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="support_is_fresh",
    category="SR结构类",
    compute_func=_compute_support_is_fresh,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑是否新鲜",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="resistance_is_fresh",
    category="SR结构类",
    compute_func=_compute_resistance_is_fresh,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力是否新鲜",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="support_is_overused",
    category="SR结构类",
    compute_func=_compute_support_is_overused,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑是否过度使用",
    direction="negative",
    is_core=False,
)

register_factor(
    name="resistance_is_overused",
    category="SR结构类",
    compute_func=_compute_resistance_is_overused,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力是否过度使用",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="support_gap_pct",
    category="SR结构类",
    compute_func=_compute_support_gap_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="flipped_support与pivot_support间距百分比",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="support_cluster_count",
    category="SR结构类",
    compute_func=_compute_support_cluster_count,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑簇内水平位数量",
    direction="positive",
    is_core=True,
)

register_factor(
    name="support_cluster_score",
    category="SR结构类",
    compute_func=_compute_support_cluster_score,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑簇综合评分",
    direction="positive",
    is_core=True,
)

register_factor(
    name="support_cluster_density",
    category="SR结构类",
    compute_func=_compute_support_cluster_density,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑簇密度",
    direction="positive",
    is_core=False,
)

register_factor(
    name="support_zone_low",
    category="SR结构类",
    compute_func=_compute_support_zone_low,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑簇下界",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="support_zone_high",
    category="SR结构类",
    compute_func=_compute_support_zone_high,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑簇上界",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="support_cluster_is_strong",
    category="SR结构类",
    compute_func=_compute_support_cluster_is_strong,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑簇是否为强簇",
    direction="positive",
    is_core=True,
)

register_factor(
    name="resistance_cluster_count",
    category="SR结构类",
    compute_func=_compute_resistance_cluster_count,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力簇内水平位数量",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="resistance_cluster_score",
    category="SR结构类",
    compute_func=_compute_resistance_cluster_score,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力簇综合评分",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="resistance_zone_low",
    category="SR结构类",
    compute_func=_compute_resistance_zone_low,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力簇下界",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="resistance_zone_high",
    category="SR结构类",
    compute_func=_compute_resistance_zone_high,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力簇上界",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="resistance_cluster_is_strong",
    category="SR结构类",
    compute_func=_compute_resistance_cluster_is_strong,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力簇是否为强簇",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="support_confluence_score",
    category="SR结构类",
    compute_func=_compute_support_confluence_score,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑共振综合评分",
    direction="positive",
    is_core=True,
)

register_factor(
    name="resistance_confluence_score",
    category="SR结构类",
    compute_func=_compute_resistance_confluence_score,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力共振综合评分",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="support_confluence_is_strong",
    category="SR结构类",
    compute_func=_compute_support_confluence_is_strong,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="支撑共振是否强共振",
    direction="positive",
    is_core=True,
)

register_factor(
    name="resistance_confluence_is_strong",
    category="SR结构类",
    compute_func=_compute_resistance_confluence_is_strong,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力共振是否强共振",
    direction="neutral",
    is_core=True,
)


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    opn = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 5000, n).astype(float)

    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    result = compute_sr_structure_factors(df)
    print(f"因子列数: {len(result.columns)}, 期望: {len(_SR_FACTOR_COLUMNS)}")
    assert list(result.columns) == _SR_FACTOR_COLUMNS, f"列名不匹配: {list(result.columns)}"
    print(result.dropna().head(5).to_string())
    print("自测通过")
