# -*- coding: utf-8 -*-
"""
Trend Script (A脚本) - 趋势判定

按照 策略1.md 规范输出:
1. trend_dir: +1 / -1 / 0
2. trend_strength: 0~1
3. trend_position: 当前位置（归一化到 0~1）
4. best_trade_zone: 最合适交易位置
   - best_zone_type: "discount" | "neutral" | "premium"
   - best_zone_score: 0~1
   - best_zone_reason: 可选

两种参考:
1. AMP的交易活跃区 (amp_plotly.py)
2. VWAP位置 (dynamic_swing_anchored_vwap.py)

Usage:
    python trend_script.py --symbol 000426 --years 1 --freq d
    python trend_script.py --symbol 000426 --years 1 --freq d --out trend.html
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

from features.amp_plotly import (
    AMPConfig,
    compute_amp_last,
    fetch_daily_pytdx,
)
from features.dynamic_swing_anchored_vwap import (
    DSAConfig,
    dynamic_swing_anchored_vwap,
    fetch_kline_pytdx,
)


@dataclass
class TrendConfig:
    tf_big: str = "d"
    amp_period: int = 200
    amp_use_log: bool = True
    amp_adaptive: bool = True
    dsa_prd: int = 50
    dsa_base_apt: float = 20.0
    dsa_use_adapt: bool = False
    strength_threshold: float = 0.3
    discount_threshold: float = 0.35
    premium_threshold: float = 0.65


@dataclass
class BestTradeZone:
    zone_type: str
    score: float
    reason: str = ""
    amp_active_upper: Optional[float] = None
    amp_active_lower: Optional[float] = None


@dataclass
class TrendOutput:
    trend_dir: int
    trend_strength: float
    trend_position: float
    best_trade_zone: BestTradeZone
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "trend_dir": self.trend_dir,
            "trend_strength": self.trend_strength,
            "trend_position": self.trend_position,
            "best_trade_zone": {
                "zone_type": self.best_trade_zone.zone_type,
                "score": self.best_trade_zone.score,
                "reason": self.best_trade_zone.reason,
                "amp_active_upper": self.best_trade_zone.amp_active_upper,
                "amp_active_lower": self.best_trade_zone.amp_active_lower,
            },
            "debug": self.debug,
        }

    def __str__(self) -> str:
        return (
            f"TrendOutput(\n"
            f"  trend_dir={self.trend_dir},\n"
            f"  trend_strength={self.trend_strength:.4f},\n"
            f"  trend_position={self.trend_position:.4f},\n"
            f"  best_trade_zone=BestTradeZone(\n"
            f"    zone_type='{self.best_trade_zone.zone_type}',\n"
            f"    score={self.best_trade_zone.score:.4f},\n"
            f"    reason='{self.best_trade_zone.reason}',\n"
            f"    amp_active_upper={self.best_trade_zone.amp_active_upper:.2f},\n"
            f"    amp_active_lower={self.best_trade_zone.amp_active_lower:.2f}\n"
            f"  )\n"
            f")"
        )


def _determine_zone_type(position: float, discount_th: float, premium_th: float) -> Tuple[str, float]:
    if position <= discount_th:
        return "discount", 1.0 - position
    elif position >= premium_th:
        return "premium", position
    else:
        dist_to_discount = abs(position - discount_th)
        dist_to_premium = abs(position - premium_th)
        dist_to_mid = abs(position - 0.5)
        min_dist = min(dist_to_discount, dist_to_premium, dist_to_mid)
        if min_dist == dist_to_discount:
            return "neutral", 0.5 + dist_to_discount
        elif min_dist == dist_to_premium:
            return "neutral", 0.5 + dist_to_premium
        else:
            return "neutral", 0.5 + dist_to_mid


def _combine_zone_scores(
    amp_zone: Tuple[str, float],
    vwap_zone: Tuple[str, float],
    amp_active_upper: Optional[float],
    amp_active_lower: Optional[float],
    amp_weight: float = 0.6,
) -> BestTradeZone:
    amp_type, amp_score = amp_zone
    vwap_type, vwap_score = vwap_zone

    if amp_type == vwap_type:
        combined_score = amp_weight * amp_score + (1 - amp_weight) * vwap_score
        return BestTradeZone(
            zone_type=amp_type,
            score=combined_score,
            reason=f"AMP和VWAP一致: {amp_type}",
            amp_active_upper=amp_active_upper,
            amp_active_lower=amp_active_lower,
        )

    if amp_type == "discount":
        if vwap_type == "neutral":
            combined_score = amp_weight * amp_score + (1 - amp_weight) * vwap_score * 0.5
            return BestTradeZone(
                zone_type="discount",
                score=combined_score,
                reason="AMP为折价区，VWAP为中性区，倾向折价",
                amp_active_upper=amp_active_upper,
                amp_active_lower=amp_active_lower,
            )
        else:
            combined_score = amp_weight * amp_score
            return BestTradeZone(
                zone_type="discount",
                score=combined_score,
                reason="AMP为折价区，VWAP为溢价区，以AMP为主",
                amp_active_upper=amp_active_upper,
                amp_active_lower=amp_active_lower,
            )

    if amp_type == "premium":
        if vwap_type == "neutral":
            combined_score = amp_weight * amp_score + (1 - amp_weight) * vwap_score * 0.5
            return BestTradeZone(
                zone_type="premium",
                score=combined_score,
                reason="AMP为溢价区，VWAP为中性区，倾向溢价",
                amp_active_upper=amp_active_upper,
                amp_active_lower=amp_active_lower,
            )
        else:
            combined_score = amp_weight * amp_score
            return BestTradeZone(
                zone_type="premium",
                score=combined_score,
                reason="AMP为溢价区，VWAP为折价区，以AMP为主",
                amp_active_upper=amp_active_upper,
                amp_active_lower=amp_active_lower,
            )

    combined_score = amp_weight * amp_score * 0.5 + (1 - amp_weight) * vwap_score
    return BestTradeZone(
        zone_type="neutral",
        score=combined_score,
        reason="综合判断为中性区",
        amp_active_upper=amp_active_upper,
        amp_active_lower=amp_active_lower,
    )


def trend_script(
    symbol: str,
    start: str,
    end: str,
    cfg: TrendConfig,
) -> TrendOutput:
    """
    趋势脚本主函数

    Args:
        symbol: 股票代码
        start: 开始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        cfg: 配置参数

    Returns:
        TrendOutput: 标准输出字段
    """
    freq = cfg.tf_big.lower()
    if freq in ("d", "day", "daily"):
        df = fetch_daily_pytdx(symbol, start, end)
    else:
        df = fetch_kline_pytdx(symbol, start, end, freq)

    if df.empty:
        raise ValueError(f"无法获取数据: symbol={symbol}, start={start}, end={end}")

    amp_cfg = AMPConfig(
        useAdaptive=cfg.amp_adaptive,
        pI=cfg.amp_period,
        uL=cfg.amp_use_log,
    )
    amp_result = compute_amp_last(df, amp_cfg)
    amp_metrics = amp_result["metrics"]

    amp_strength = amp_metrics["strength_pR_cD"]
    amp_strength_normalized = (amp_strength + 1.0) / 2.0
    amp_position = amp_metrics["close_pos_0_1"]

    amp_zone = _determine_zone_type(
        amp_position,
        cfg.discount_threshold,
        cfg.premium_threshold,
    )

    dsa_cfg = DSAConfig(
        prd=cfg.dsa_prd,
        baseAPT=cfg.dsa_base_apt,
        useAdapt=cfg.dsa_use_adapt,
    )
    vwap_series, dir_series, pivot_labels, segments = dynamic_swing_anchored_vwap(df, dsa_cfg)

    last_close = float(df["close"].iloc[-1])
    last_vwap = float(vwap_series.iloc[-1]) if pd.notna(vwap_series.iloc[-1]) else last_close
    last_dir = int(dir_series.iloc[-1]) if pd.notna(dir_series.iloc[-1]) else 0

    vwap_position = 0.5
    if last_vwap > 0:
        recent_high = float(df["high"].tail(50).max())
        recent_low = float(df["low"].tail(50).min())
        price_range = recent_high - recent_low
        if price_range > 0:
            vwap_position = (last_vwap - recent_low) / price_range
            vwap_position = float(np.clip(vwap_position, 0.0, 1.0))

    vwap_zone = _determine_zone_type(
        vwap_position,
        cfg.discount_threshold,
        cfg.premium_threshold,
    )

    activity_lines = amp_result.get("activityLines", [])
    amp_active_upper = None
    amp_active_lower = None
    if activity_lines:
        end_y_values = []
        for line in activity_lines:
            if len(line["y"]) >= 2:
                end_y_values.append(line["y"][-1])
        if end_y_values:
            amp_active_upper = float(np.max(end_y_values))
            amp_active_lower = float(np.min(end_y_values))

    combined_zone = _combine_zone_scores(
        amp_zone, vwap_zone, amp_active_upper, amp_active_lower, amp_weight=0.6
    )

    if amp_strength_normalized >= cfg.strength_threshold:
        if last_dir > 0:
            trend_dir = 1
        elif last_dir < 0:
            trend_dir = -1
        else:
            if amp_position < 0.5:
                trend_dir = 1
            elif amp_position > 0.5:
                trend_dir = -1
            else:
                trend_dir = 0
    else:
        trend_dir = 0

    active_lines = amp_metrics.get("active_lines_stats", [])
    active_line_info = ""
    if active_lines:
        for line in active_lines[:2]:
            active_line_info += (
                f"线{line['rank']}: 高度占比={line['height_end_0_1']:.3f}, "
                f"活跃度={line['count_frac']:.3f}; "
            )

    debug_info = {
        "symbol": symbol,
        "data_range": f"{df.index[0]} ~ {df.index[-1]}",
        "bars": len(df),
        "amp": {
            "window_len": amp_metrics["window_len"],
            "final_period": amp_metrics["finalPeriod"],
            "strength_pR": amp_metrics["strength_pR_cD"],
            "strength_normalized": amp_strength_normalized,
            "close_pos_0_1": amp_position,
            "upper_end": amp_metrics["upper_end"],
            "lower_end": amp_metrics["lower_end"],
            "active_upper": amp_active_upper,
            "active_lower": amp_active_lower,
            "zone_type": amp_zone[0],
            "zone_score": amp_zone[1],
            "active_lines": active_line_info,
        },
        "vwap": {
            "last_vwap": last_vwap,
            "last_close": last_close,
            "last_dir": last_dir,
            "vwap_position": vwap_position,
            "zone_type": vwap_zone[0],
            "zone_score": vwap_zone[1],
        },
        "combined_zone": {
            "zone_type": combined_zone.zone_type,
            "score": combined_zone.score,
            "reason": combined_zone.reason,
        },
    }

    return TrendOutput(
        trend_dir=trend_dir,
        trend_strength=amp_strength_normalized,
        trend_position=amp_position,
        best_trade_zone=combined_zone,
        debug=debug_info,
    )


def build_trend_html(
    df: pd.DataFrame,
    amp_result: Dict,
    vwap_series: pd.Series,
    dir_series: pd.Series,
    output: TrendOutput,
    out_path: str,
) -> None:
    """生成HTML可视化报告"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    amp_metrics = amp_result["metrics"]
    ch = amp_result["channel"]
    cfg = amp_result["cfg"]

    x_str = [str(idx) for idx in df.index]
    ch_x_str = [str(x) for x in ch["x"]]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=x_str,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ch_x_str,
            y=ch["upper"],
            mode="lines",
            line=dict(color=cfg.regColor, width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ch_x_str,
            y=ch["lower"],
            mode="lines",
            line=dict(color=cfg.regColor, width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ch_x_str,
            y=ch["upper"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ch_x_str,
            y=ch["lower"],
            mode="lines",
            fill="tonexty",
            fillcolor=cfg.channelFill,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )

    for al in amp_result.get("activityLines", []):
        al_x_str = [str(x) for x in al["x"]]
        fig.add_trace(
            go.Scatter(
                x=al_x_str,
                y=al["y"],
                mode="lines",
                line=dict(color=al["color"], width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1,
        )

    shapes = []
    for poly in amp_result.get("profilePolys", []):
        x0 = str(poly["x0"])
        x1 = str(poly["x1"])
        y0t = poly["y0_top"]
        y0b = poly["y0_bot"]
        y1t = poly["y1_top"]
        y1b = poly["y1_bot"]
        shapes.append(dict(
            type="path",
            xref="x", yref="y",
            path=f"M {x0} {y0t} L {x0} {y0b} L {x1} {y1b} L {x1} {y1t} Z",
            fillcolor=poly["fill"],
            line=dict(width=0),
            layer="below",
            opacity=1.0,
        ))

    vwap_x_str = [str(idx) for idx in vwap_series.index]
    fig.add_trace(
        go.Scatter(
            x=vwap_x_str,
            y=vwap_series,
            mode="lines",
            line=dict(color="rgba(255,23,68,0.7)", width=2),
            name="VWAP",
        ),
        row=1, col=1,
    )

    vol_colors = np.where(df["close"] >= df["open"], "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
    fig.add_trace(
        go.Bar(
            x=x_str,
            y=df["volume"],
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2, col=1,
    )

    dir_text = {1: "↑ 上涨", -1: "↓ 下跌", 0: "→ 震荡"}[output.trend_dir]

    info_html = (
        f"<b>Trend Script 输出</b><br>"
        f"趋势方向: {dir_text}<br>"
        f"趋势强度: {output.trend_strength:.4f}<br>"
        f"当前位置: {output.trend_position:.4f}<br>"
        f"<br><b>最佳交易区</b><br>"
        f"类型: {output.best_trade_zone.zone_type}<br>"
        f"得分: {output.best_trade_zone.score:.4f}<br>"
        f"原因: {output.best_trade_zone.reason}<br>"
        f"活跃区上界: {output.best_trade_zone.amp_active_upper:.2f}<br>"
        f"活跃区下界: {output.best_trade_zone.amp_active_lower:.2f}<br>"
        f"<br><b>AMP指标</b><br>"
        f"窗口长度: {amp_metrics['window_len']}<br>"
        f"强度(pR): {amp_metrics['strength_pR_cD']:.4f}<br>"
        f"通道位置: {amp_metrics['close_pos_0_1']:.4f}<br>"
        f"<br><b>VWAP指标</b><br>"
        f"最新VWAP: {output.debug['vwap']['last_vwap']:.2f}<br>"
        f"VWAP方向: {output.debug['vwap']['last_dir']}<br>"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        text=info_html,
        align="left",
        showarrow=False,
        font=dict(size=12, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=8,
    )

    fig.add_annotation(
        x=ch_x_str[0],
        y=amp_metrics["lower_start"],
        xref="x",
        yref="y",
        text=f"{amp_metrics['strength_pR_cD']:.3f}",
        showarrow=False,
        font=dict(size=12, color="rgba(200,200,200,0.9)"),
        bgcolor="rgba(0,0,0,0.25)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=4,
        yshift=20,
    )

    fig.update_layout(
        title=f"Trend Script - {output.debug['symbol']}",
        shapes=shapes,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=900,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="000426", help="股票代码")
    ap.add_argument("--years", type=int, default=1, help="数据年限")
    ap.add_argument("--freq", type=str, default="d", help="周期: d(日线)")
    ap.add_argument("--out", type=str, default="", help="输出HTML路径(可选)")

    ap.add_argument("--amp-period", type=int, default=200, help="AMP周期")
    ap.add_argument("--amp-use-log", action="store_true", default=True, help="AMP使用对数计算")
    ap.add_argument("--amp-no-adaptive", action="store_true", help="AMP禁用自适应周期")
    ap.add_argument("--strength-threshold", type=float, default=0.3, help="强度阈值")
    ap.add_argument("--discount-threshold", type=float, default=0.35, help="折价区阈值")
    ap.add_argument("--premium-threshold", type=float, default=0.65, help="溢价区阈值")

    args = ap.parse_args()

    end = datetime.now().date()
    start = end - timedelta(days=365 * args.years)

    cfg = TrendConfig(
        tf_big=args.freq,
        amp_period=args.amp_period,
        amp_use_log=args.amp_use_log,
        amp_adaptive=not args.amp_no_adaptive,
        strength_threshold=args.strength_threshold,
        discount_threshold=args.discount_threshold,
        premium_threshold=args.premium_threshold,
    )

    print(f"\n{'='*60}")
    print(f"Trend Script (A脚本) - 趋势判定")
    print(f"{'='*60}")
    print(f"股票代码: {args.symbol}")
    print(f"时间范围: {start} ~ {end}")
    print(f"周期: {args.freq}")
    print(f"{'='*60}\n")

    output = trend_script(args.symbol, str(start), str(end), cfg)

    print(f"\n{'='*60}")
    print(f"输出结果:")
    print(f"{'='*60}")
    print(output)
    print(f"\n详细调试信息:")
    print(f"{'='*60}")
    for key, value in output.debug.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    if args.out:
        freq = args.freq.lower()
        if freq in ("d", "day", "daily"):
            df = fetch_daily_pytdx(args.symbol, str(start), str(end))
        else:
            df = fetch_kline_pytdx(args.symbol, str(start), str(end), freq)

        amp_cfg = AMPConfig(
            useAdaptive=cfg.amp_adaptive,
            pI=cfg.amp_period,
            uL=cfg.amp_use_log,
        )
        amp_result = compute_amp_last(df, amp_cfg)

        dsa_cfg = DSAConfig(
            prd=cfg.dsa_prd,
            baseAPT=cfg.dsa_base_apt,
            useAdapt=cfg.dsa_use_adapt,
        )
        vwap_series, dir_series, _, _ = dynamic_swing_anchored_vwap(df, dsa_cfg)

        build_trend_html(df, amp_result, vwap_series, dir_series, output, args.out)


if __name__ == "__main__":
    main()
