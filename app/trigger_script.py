# -*- coding: utf-8 -*-
"""
Trigger Script (C脚本) - 触发点判定

按照 策略1.md 规范输出:
1. 每周期信号:
   - bull_div[s]: 底背离 True/False
   - bear_div[s]: 顶背离 True/False
   - div_age[s]: 0~1 (越小越新)
   - signal_quality[s]: 0~1
2. 最终触发选择:
   - buy_trigger_tf: 第一个满足买触发条件的tf
   - sell_trigger_tf: 第一个满足卖触发条件的tf
   - buy_trigger: bool
   - sell_trigger: bool

Usage:
    python src/trigger_script.py --symbol 000426 --tf-small 5,15,30 --years 1
    python src/trigger_script.py --symbol 000426 --tf-small 5,15,30 --years 1 --out trigger.html
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/zhenbao/Nextcloud/coding/交易")

from features.divergence_many_plotly import (
    DivConfig,
    run_divergence_engine,
    fetch_bars_pytdx,
)


@dataclass
class TriggerConfig:
    tf_small: List[str] = field(default_factory=lambda: ["5", "15", "30"])
    small_priority: List[str] = field(default_factory=lambda: ["5", "15", "30"])
    prd: int = 5
    source: str = "Close"
    searchdiv: str = "Regular"
    maxpp: int = 10
    maxbars: int = 100
    age_th: float = 0.3
    calcmacd: bool = True
    calcmacda: bool = True
    calcrsi: bool = False
    calcstoc: bool = False
    calccci: bool = False
    calcmom: bool = False
    calcobv: bool = True
    calcvwmacd: bool = False
    calccmf: bool = False
    calcmfi: bool = False
    calcext: bool = False


@dataclass
class TFSignal:
    bull_div: bool = False
    bear_div: bool = False
    div_age: float = 1.0
    signal_quality: float = 0.0
    div_len: int = 0
    div_type: str = "none"


@dataclass
class TriggerOutput:
    signals: Dict[str, TFSignal] = field(default_factory=dict)
    buy_trigger_tf: Optional[str] = None
    sell_trigger_tf: Optional[str] = None
    buy_trigger: bool = False
    sell_trigger: bool = False
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "signals": {tf: {
                "bull_div": s.bull_div,
                "bear_div": s.bear_div,
                "div_age": s.div_age,
                "signal_quality": s.signal_quality,
            } for tf, s in self.signals.items()},
            "buy_trigger_tf": self.buy_trigger_tf,
            "sell_trigger_tf": self.sell_trigger_tf,
            "buy_trigger": self.buy_trigger,
            "sell_trigger": self.sell_trigger,
            "debug": self.debug,
        }

    def __str__(self) -> str:
        lines = ["TriggerOutput("]
        for tf, s in self.signals.items():
            bull = "✓" if s.bull_div else "✗"
            bear = "✓" if s.bear_div else "✗"
            lines.append(f"  {tf}: bull_div={bull}, bear_div={bear}, age={s.div_age:.3f}, quality={s.signal_quality:.3f}")
        lines.append(f"  buy_trigger_tf={self.buy_trigger_tf}")
        lines.append(f"  sell_trigger_tf={self.sell_trigger_tf}")
        lines.append(f"  buy_trigger={self.buy_trigger}")
        lines.append(f"  sell_trigger={self.sell_trigger}")
        lines.append(")")
        return "\n".join(lines)


def _detect_divergence_for_tf(
    df: pd.DataFrame,
    cfg: TriggerConfig,
) -> TFSignal:
    """检测单个周期的背离信号 - 直接使用 run_divergence_engine"""
    if len(df) < cfg.prd * 3 + 10:
        return TFSignal()

    div_cfg = DivConfig(
        prd=cfg.prd,
        source=cfg.source,
        searchdiv=cfg.searchdiv,
        maxpp=cfg.maxpp,
        maxbars=cfg.maxbars,
        calcmacd=cfg.calcmacd,
        calcmacda=cfg.calcmacda,
        calcrsi=cfg.calcrsi,
        calcstoc=cfg.calcstoc,
        calccci=cfg.calccci,
        calcmom=cfg.calcmom,
        calcobv=cfg.calcobv,
        calcvwmacd=cfg.calcvwmacd,
        calccmf=cfg.calccmf,
        calcmfi=cfg.calcmfi,
        calcext=cfg.calcext,
    )

    pos_lines, neg_lines, pos_labels, neg_labels, last_feat = run_divergence_engine(df, div_cfg)

    bull_div = False
    bear_div = False
    min_age = 1.0
    max_quality = 0.0
    best_div_len = 0
    best_div_type = "none"

    div_last_type = last_feat.get("div_last_type", "none")
    div_last_age = last_feat.get("div_last_age", 999)
    div_bias = last_feat.get("div_bias", 0.0)

    if div_last_type == "bottom":
        bull_div = True
        age = min(div_last_age / cfg.maxbars, 1.0)
        quality = 1.0 - age
        max_quality = quality
        min_age = age
        best_div_len = div_last_age
        best_div_type = "pos_reg"
    elif div_last_type == "top":
        bear_div = True
        age = min(div_last_age / cfg.maxbars, 1.0)
        quality = 1.0 - age
        max_quality = quality
        min_age = age
        best_div_len = div_last_age
        best_div_type = "neg_reg"
    elif div_last_type == "conflict":
        bull_div = True
        bear_div = True
        age = min(div_last_age / cfg.maxbars, 1.0)
        quality = 1.0 - age
        max_quality = quality
        min_age = age
        best_div_len = div_last_age
        best_div_type = "conflict"

    return TFSignal(
        bull_div=bull_div,
        bear_div=bear_div,
        div_age=min_age,
        signal_quality=max_quality,
        div_len=best_div_len,
        div_type=best_div_type,
    )


def trigger_script(
    symbol: str,
    start: str,
    end: str,
    cfg: TriggerConfig,
) -> TriggerOutput:
    """
    触发脚本主函数

    Args:
        symbol: 股票代码
        start: 开始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        cfg: 配置参数

    Returns:
        TriggerOutput: 标准输出字段
    """
    signals: Dict[str, TFSignal] = {}
    debug_info: Dict = {
        "symbol": symbol,
        "data_range": f"{start} ~ {end}",
        "tf_small": cfg.tf_small,
        "small_priority": cfg.small_priority,
        "age_th": cfg.age_th,
        "tf_details": {},
    }

    for tf in cfg.tf_small:
        df = fetch_bars_pytdx(symbol, tf, start, end)
        if df.empty:
            signals[tf] = TFSignal()
            debug_info["tf_details"][tf] = {"bars": 0, "error": "无数据"}
            continue

        signal = _detect_divergence_for_tf(df, cfg)
        signals[tf] = signal

        debug_info["tf_details"][tf] = {
            "bars": len(df),
            "bull_div": signal.bull_div,
            "bear_div": signal.bear_div,
            "div_age": signal.div_age,
            "signal_quality": signal.signal_quality,
            "div_len": signal.div_len,
            "div_type": signal.div_type,
        }

    buy_trigger_tf = None
    sell_trigger_tf = None

    for tf in cfg.small_priority:
        if tf not in signals:
            continue
        s = signals[tf]

        if buy_trigger_tf is None and s.bull_div and s.div_age <= cfg.age_th:
            buy_trigger_tf = tf

        if sell_trigger_tf is None and s.bear_div and s.div_age <= cfg.age_th:
            sell_trigger_tf = tf

    output = TriggerOutput(
        signals=signals,
        buy_trigger_tf=buy_trigger_tf,
        sell_trigger_tf=sell_trigger_tf,
        buy_trigger=(buy_trigger_tf is not None),
        sell_trigger=(sell_trigger_tf is not None),
        debug=debug_info,
    )

    return output


def build_trigger_html(
    output: TriggerOutput,
    cfg: TriggerConfig,
    out_path: str,
) -> None:
    """生成HTML可视化报告"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=1,
    )

    tf_list = list(cfg.small_priority)
    y_pos = list(range(len(tf_list)))

    colors_buy = []
    colors_sell = []
    texts = []

    for tf in tf_list:
        s = output.signals.get(tf, TFSignal())
        if s.bull_div and s.div_age <= cfg.age_th:
            colors_buy.append("rgba(0,230,118,0.9)")
        elif s.bull_div:
            colors_buy.append("rgba(0,230,118,0.4)")
        else:
            colors_buy.append("rgba(100,100,100,0.3)")

        if s.bear_div and s.div_age <= cfg.age_th:
            colors_sell.append("rgba(255,23,68,0.9)")
        elif s.bear_div:
            colors_sell.append("rgba(255,23,68,0.4)")
        else:
            colors_sell.append("rgba(100,100,100,0.3)")

        texts.append(f"age={s.div_age:.2f}<br>quality={s.signal_quality:.2f}")

    fig.add_trace(go.Bar(
        x=[1] * len(tf_list),
        y=y_pos,
        orientation="h",
        marker_color=colors_buy,
        text=[f"底背离 {tf}" for tf in tf_list],
        textposition="inside",
        showlegend=False,
        name="底背离",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=[1] * len(tf_list),
        y=[y + 0.3 for y in y_pos],
        orientation="h",
        marker_color=colors_sell,
        text=[f"顶背离 {tf}" for tf in tf_list],
        textposition="inside",
        showlegend=False,
        name="顶背离",
    ), row=1, col=1)

    buy_text = f"买触发: {output.buy_trigger_tf}" if output.buy_trigger_tf else "买触发: 无"
    sell_text = f"卖触发: {output.sell_trigger_tf}" if output.sell_trigger_tf else "卖触发: 无"

    info_html = (
        f"<b>Trigger Script 输出</b><br>"
        f"<br><b>最终触发</b><br>"
        f"{buy_text}<br>"
        f"{sell_text}<br>"
        f"<br><b>配置</b><br>"
        f"周期优先级: {cfg.small_priority}<br>"
        f"年龄阈值: {cfg.age_th}<br>"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        text=info_html,
        align="left",
        showarrow=False,
        font=dict(size=12, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=8,
    )

    fig.update_layout(
        title=f"Trigger Script - {output.debug['symbol']}",
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickmode="array",
            tickvals=y_pos,
            ticktext=tf_list,
        ),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="000426", help="股票代码")
    ap.add_argument("--years", type=int, default=1, help="数据年限")
    ap.add_argument("--tf-small", type=str, default="5,15,30", help="小周期列表(逗号分隔)")
    ap.add_argument("--priority", type=str, default="", help="周期优先级(逗号分隔), 默认同tf-small")
    ap.add_argument("--out", type=str, default="", help="输出HTML路径(可选)")

    ap.add_argument("--prd", type=int, default=5, help="Pivot周期")
    ap.add_argument("--source", type=str, default="Close", choices=["Close", "High/Low"], help="Pivot源")
    ap.add_argument("--searchdiv", type=str, default="Regular", choices=["Regular", "Hidden", "Regular/Hidden"], help="背离类型")
    ap.add_argument("--age-th", type=float, default=0.3, help="年龄阈值(0~1)")

    args = ap.parse_args()

    end = datetime.now().date()
    start = end - timedelta(days=365 * args.years)

    tf_small = [x.strip() for x in args.tf_small.split(",")]
    priority = [x.strip() for x in args.priority.split(",")] if args.priority else tf_small

    cfg = TriggerConfig(
        tf_small=tf_small,
        small_priority=priority,
        prd=args.prd,
        source=args.source,
        searchdiv=args.searchdiv,
        age_th=args.age_th,
    )

    print(f"\n{'='*60}")
    print(f"Trigger Script (C脚本) - 触发点判定")
    print(f"{'='*60}")
    print(f"股票代码: {args.symbol}")
    print(f"时间范围: {start} ~ {end}")
    print(f"小周期: {tf_small}")
    print(f"优先级: {priority}")
    print(f"{'='*60}\n")

    output = trigger_script(args.symbol, str(start), str(end), cfg)

    print(f"\n{'='*60}")
    print(f"输出结果:")
    print(f"{'='*60}")
    print(output)

    print(f"\n详细调试信息:")
    print(f"{'='*60}")
    for key, value in output.debug.items():
        if key == "tf_details":
            print(f"{key}:")
            for tf, details in value.items():
                print(f"  {tf}: {details}")
        else:
            print(f"{key}: {value}")

    if args.out:
        build_trigger_html(output, cfg, args.out)


if __name__ == "__main__":
    main()
