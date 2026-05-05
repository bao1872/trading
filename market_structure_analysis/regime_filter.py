"""
regime_filter.py — 市场环境过滤器

Purpose:  根据 v2 五档标签为策略提供环境过滤能力，把"解释工具"变成"交易工具"。
Inputs:   已落盘的 daily_summary CSV（market_state_classifier 输出），含 main_label_v2 列
Outputs:  过滤后的策略信号、仓位建议
How to Run:
    python market_structure_analysis/regime_filter.py
Examples:
    from market_structure_analysis.regime_filter import load_regime_series, get_regime_rules
    regime = load_regime_series("2024-01-01", "2024-12-31")
    rules = get_regime_rules(regime.get("2024-06-15"))
Side Effects: 无（只读 CSV 文件）
"""

import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REGIME_RULES: Dict[str, Dict] = {
    "强扩张": {
        "allow_aggressive": True,
        "allow_breakout": True,
        "allow_reversal": True,
        "position": "full",
        "position_pct": 1.0,
        "description": "允许进攻型策略，可放宽阈值",
    },
    "弱扩张": {
        "allow_aggressive": False,
        "allow_breakout": True,
        "allow_reversal": True,
        "position": "normal",
        "position_pct": 0.8,
        "description": "可以做多，但偏向前排/强确认信号",
    },
    "中性": {
        "allow_aggressive": False,
        "allow_breakout": True,
        "allow_reversal": True,
        "position": "reduced",
        "position_pct": 0.5,
        "description": "控制仓位，少追高",
    },
    "弱退潮": {
        "allow_aggressive": False,
        "allow_breakout": False,
        "allow_reversal": True,
        "position": "cautious",
        "position_pct": 0.3,
        "description": "谨慎，只做低风险修复",
    },
    "强退潮": {
        "allow_aggressive": False,
        "allow_breakout": False,
        "allow_reversal": False,
        "position": "minimal",
        "position_pct": 0.0,
        "description": "原则上停做高弹性追涨",
    },
}

DEFAULT_CSV_PATH = "/tmp/market_daily_summary.csv"
_global_regime_series: Optional[pd.Series] = None


def load_regime_series(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    csv_path: Optional[str] = None,
    use_cache: bool = True,
) -> pd.Series:
    """
    从 CSV 加载 v2 标签序列。

    Parameters
    ----------
    start_date : str, optional
    end_date : str, optional
    csv_path : str, optional
        默认 DEFAULT_CSV_PATH
    use_cache : bool
        是否使用全局缓存

    Returns
    -------
    pd.Series
        index=日期, values=main_label_v2
    """
    global _global_regime_series

    if use_cache and _global_regime_series is not None:
        result = _global_regime_series
        if start_date is not None:
            result = result[result.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result[result.index <= pd.Timestamp(end_date)]
        return result

    path = csv_path or DEFAULT_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"标签 CSV 不存在: {path}，请先运行 market_state_classifier.py --output {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "main_label_v2" not in df.columns:
        raise KeyError("CSV 缺少 main_label_v2 列")

    regime = df["main_label_v2"].dropna()

    if start_date is not None:
        regime = regime[regime.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        regime = regime[regime.index <= pd.Timestamp(end_date)]

    if use_cache:
        _global_regime_series = df["main_label_v2"].dropna()

    return regime


def get_regime(date) -> Dict:
    """
    获取指定日期的完整环境信息。

    Parameters
    ----------
    date : str or pd.Timestamp

    Returns
    -------
    dict
        {main_label_v2, size_style_label, breadth_label, confidence, ...}
    """
    path = DEFAULT_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"标签 CSV 不存在: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    dt = pd.Timestamp(date)

    if dt not in df.index:
        closest = df.index[df.index <= dt]
        if closest.empty:
            return {"main_label_v2": "未知", "position": "unknown", "position_pct": 0.0}
        dt = closest[-1]

    row = df.loc[dt]
    result = {
        "trade_date": dt,
        "main_label_v2": row.get("main_label_v2", "未知"),
        "size_style_label": row.get("size_style_label", "未知"),
        "breadth_label": row.get("breadth_label", "未知"),
        "confidence": row.get("confidence", 0.0),
        "dominant_driver": row.get("dominant_driver", "未知"),
        "secondary_driver": row.get("secondary_driver", "未知"),
        "risk_note": row.get("risk_note", ""),
    }
    result.update(get_regime_rules(result["main_label_v2"]))
    return result


def get_regime_rules(label: str) -> Dict:
    """获取指定标签的交易规则。"""
    return REGIME_RULES.get(label, REGIME_RULES["中性"]).copy()


def apply_regime_filter(
    signals_df: pd.DataFrame,
    regime_label: str,
    signal_type_col: str = "signal_type",
) -> pd.DataFrame:
    """
    根据当前环境标签过滤策略信号。

    Parameters
    ----------
    signals_df : pd.DataFrame
        策略信号表，需含 signal_type_col 列（值为 aggressive/breakout/reversal）
    regime_label : str
        当日 v2 标签
    signal_type_col : str
        信号类型列名

    Returns
    -------
    pd.DataFrame
        过滤后的信号
    """
    rules = get_regime_rules(regime_label)

    if signal_type_col not in signals_df.columns:
        logger.warning("信号表无 %s 列，跳过过滤", signal_type_col)
        return signals_df

    mask = pd.Series(True, index=signals_df.index)
    if not rules["allow_aggressive"]:
        mask = mask & (signals_df[signal_type_col] != "aggressive")
    if not rules["allow_breakout"]:
        mask = mask & (signals_df[signal_type_col] != "breakout")
    if not rules["allow_reversal"]:
        mask = mask & (signals_df[signal_type_col] != "reversal")

    return signals_df[mask]


def generate_regime_summary(regime_series: pd.Series) -> str:
    """生成环境统计摘要文本。"""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"市场环境统计 ({len(regime_series)} 个交易日)")
    lines.append(f"")

    counts = regime_series.value_counts()
    total = len(regime_series)
    for label in ["强扩张", "弱扩张", "中性", "弱退潮", "强退潮"]:
        cnt = counts.get(label, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        lines.append(f"  {label:6s}: {cnt:4d} ({pct:5.1f}%) {bar}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="市场环境过滤器")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="标签 CSV 路径")
    parser.add_argument("--date", type=str, default=None, help="查询指定日期的环境（如 2024-06-15）")
    parser.add_argument("--summary", action="store_true", help="输出全量环境统计")
    args = parser.parse_args()

    if args.date:
        info = get_regime(args.date)
        print(f"日期: {info['trade_date'].date() if hasattr(info['trade_date'], 'date') else info['trade_date']}")
        print(f"主标签: {info['main_label_v2']}")
        print(f"风格: {info['size_style_label']}")
        print(f"广度: {info['breadth_label']}")
        print(f"仓位建议: {info['position']} ({info['position_pct']*100:.0f}%)")
        print(f"描述: {info['description']}")
        if info.get("risk_note"):
            print(f"风险: {info['risk_note']}")

    if args.summary:
        regime = load_regime_series(csv_path=args.csv, use_cache=False)
        print(generate_regime_summary(regime))


if __name__ == "__main__":
    main()
