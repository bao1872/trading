# -*- coding: utf-8 -*-
"""
Space Script (B脚本) - 空间判定

按照 策略1.md 规范输出:
1. upper_bound: 最终上界
2. lower_bound: 最终下界
3. stop_price: 止损价格
4. reward: 盈利空间
5. risk: 风险空间
6. rr: 盈亏比
7. space_left: 剩余空间
8. allow_takeprofit: 是否允许止盈

两套空间参考:
1. Liquidity Zones (流动性区域)
2. Bollinger Bands 20周期 (布林带)

Usage:
    python src/space_script.py --symbol 000426 --freq 60 --years 1
    python src/space_script.py --symbol 000426 --freq 60 --years 1 --out space.html
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

from cores.liquidity_zones_plotly import (
    LZConfig,
    build_liquidity_zones,
    fetch_kline_pytdx,
    atr_wilder,
)


@dataclass
class SpaceConfig:
    tf_mid: str = "60"
    lz_left_bars: int = 10
    lz_qty_pivots: int = 10
    lz_flt: str = "Mid"
    lz_dynamic: bool = False
    bb_period: int = 20
    bb_std: float = 2.0
    sl_buffer_atr: float = 0.5
    rr_min: float = 1.5
    space_th: float = 1.0


@dataclass
class SpaceZone:
    upper_bound: float
    lower_bound: float
    upper_src: str
    lower_src: str
    stop_price: Optional[float]
    reward: float
    risk: float
    rr: float
    rr_pass: bool
    space_left: float
    space_left_atr: float
    allow_takeprofit: bool

    def to_dict(self) -> Dict:
        return {
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "upper_src": self.upper_src,
            "lower_src": self.lower_src,
            "stop_price": self.stop_price,
            "reward": self.reward,
            "risk": self.risk,
            "rr": self.rr,
            "rr_pass": self.rr_pass,
            "space_left": self.space_left,
            "space_left_atr": self.space_left_atr,
            "allow_takeprofit": self.allow_takeprofit,
        }

    def __str__(self) -> str:
        return (
            f"SpaceZone(\n"
            f"  upper_bound={self.upper_bound:.2f} ({self.upper_src}),\n"
            f"  lower_bound={self.lower_bound:.2f} ({self.lower_src}),\n"
            f"  stop_price={self.stop_price:.2f},\n"
            f"  reward={self.reward:.2f},\n"
            f"  risk={self.risk:.2f},\n"
            f"  rr={self.rr:.2f},\n"
            f"  rr_pass={self.rr_pass},\n"
            f"  space_left={self.space_left:.2f},\n"
            f"  space_left_atr={self.space_left_atr:.2f},\n"
            f"  allow_takeprofit={self.allow_takeprofit}\n"
            f")"
        )


@dataclass
class SpaceOutput:
    lz_space: SpaceZone
    bb_space: SpaceZone
    lz_zones: List[Dict] = field(default_factory=list)
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "lz_space": self.lz_space.to_dict(),
            "bb_space": self.bb_space.to_dict(),
            "debug": self.debug,
        }

    def __str__(self) -> str:
        return (
            f"SpaceOutput(\n"
            f"=== Liquidity Zones Space ===\n"
            f"{self.lz_space}\n"
            f"=== Bollinger Bands Space ===\n"
            f"{self.bb_space}\n"
            f")"
        )


def _calc_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带"""
    mid = df["close"].rolling(period, min_periods=period).mean()
    std = df["close"].rolling(period, min_periods=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def _get_lz_bounds(
    df: pd.DataFrame,
    cfg: SpaceConfig,
) -> Tuple[Optional[float], Optional[float], List[Dict]]:
    """从Liquidity Zones获取上下界"""
    lz_cfg = LZConfig(
        leftBars=cfg.lz_left_bars,
        qty_pivots=cfg.lz_qty_pivots,
        flt=cfg.lz_flt,
        dynamic=cfg.lz_dynamic,
        hidePivot=True,
    )
    payload = build_liquidity_zones(df, lz_cfg)
    zones = payload["zones"]

    last_close = float(df["close"].iloc[-1])

    upper_zones = [
        z for z in zones
        if z["kind"] == "upper" and not z["grabbed"] and z["y"] > last_close
    ]
    lower_zones = [
        z for z in zones
        if z["kind"] == "lower" and not z["grabbed"] and z["y"] < last_close
    ]

    nearest_upper = None
    if upper_zones:
        nearest_upper = min(upper_zones, key=lambda z: z["y"] - last_close)
        nearest_upper = nearest_upper["y"]

    nearest_lower = None
    if lower_zones:
        nearest_lower = max(lower_zones, key=lambda z: z["y"])
        nearest_lower = nearest_lower["y"]

    return nearest_upper, nearest_lower, zones


def _calc_space_metrics(
    upper_bound: Optional[float],
    lower_bound: Optional[float],
    entry_price: float,
    atr: float,
    sl_buffer_atr: float,
    rr_min: float,
    space_th: float,
    src_upper: str,
    src_lower: str,
) -> SpaceZone:
    """计算空间指标"""
    if upper_bound is None or lower_bound is None:
        return SpaceZone(
            upper_bound=upper_bound or 0.0,
            lower_bound=lower_bound or 0.0,
            upper_src=src_upper,
            lower_src=src_lower,
            stop_price=None,
            reward=0.0,
            risk=0.0,
            rr=0.0,
            rr_pass=False,
            space_left=0.0,
            space_left_atr=0.0,
            allow_takeprofit=False,
        )

    stop_price = lower_bound - sl_buffer_atr * atr
    reward = upper_bound - entry_price
    risk = entry_price - stop_price

    if risk > 0 and reward > 0:
        rr = reward / risk
    else:
        rr = 0.0

    rr_pass = (rr >= rr_min) and (reward > 0) and (risk > 0)

    space_left = upper_bound - entry_price
    space_left_atr = space_left / atr if atr > 0 else 0.0
    allow_takeprofit = space_left_atr <= space_th

    return SpaceZone(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        upper_src=src_upper,
        lower_src=src_lower,
        stop_price=stop_price,
        reward=reward,
        risk=risk,
        rr=rr,
        rr_pass=rr_pass,
        space_left=space_left,
        space_left_atr=space_left_atr,
        allow_takeprofit=allow_takeprofit,
    )


def space_script(
    symbol: str,
    start: str,
    end: str,
    cfg: SpaceConfig,
) -> SpaceOutput:
    """
    空间脚本主函数

    Args:
        symbol: 股票代码
        start: 开始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        cfg: 配置参数

    Returns:
        SpaceOutput: 标准输出字段
    """
    freq = cfg.tf_mid
    df = fetch_kline_pytdx(symbol, start, end, freq)

    if df.empty:
        raise ValueError(f"无法获取数据: symbol={symbol}, start={start}, end={end}")

    last_close = float(df["close"].iloc[-1])
    atr = atr_wilder(df, 200).iloc[-1]
    if pd.isna(atr):
        atr = atr_wilder(df, 50).iloc[-1]
    if pd.isna(atr):
        atr = 0.0

    lz_upper, lz_lower, lz_zones = _get_lz_bounds(df, cfg)

    lz_space = _calc_space_metrics(
        upper_bound=lz_upper,
        lower_bound=lz_lower,
        entry_price=last_close,
        atr=atr,
        sl_buffer_atr=cfg.sl_buffer_atr,
        rr_min=cfg.rr_min,
        space_th=cfg.space_th,
        src_upper="LZ",
        src_lower="LZ",
    )

    bb_upper, bb_mid, bb_lower = _calc_bollinger_bands(df, cfg.bb_period, cfg.bb_std)
    bb_upper_last = float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else None
    bb_lower_last = float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else None

    bb_space = _calc_space_metrics(
        upper_bound=bb_upper_last,
        lower_bound=bb_lower_last,
        entry_price=last_close,
        atr=atr,
        sl_buffer_atr=cfg.sl_buffer_atr,
        rr_min=cfg.rr_min,
        space_th=cfg.space_th,
        src_upper=f"BB{cfg.bb_period}",
        src_lower=f"BB{cfg.bb_period}",
    )

    active_lz_zones = [z for z in lz_zones if not z["grabbed"]]
    upper_zones_info = [
        {"y": z["y"], "pivot_time": str(z["pivot_time"])}
        for z in active_lz_zones if z["kind"] == "upper" and z["y"] > last_close
    ]
    lower_zones_info = [
        {"y": z["y"], "pivot_time": str(z["pivot_time"])}
        for z in active_lz_zones if z["kind"] == "lower" and z["y"] < last_close
    ]

    debug_info = {
        "symbol": symbol,
        "data_range": f"{df.index[0]} ~ {df.index[-1]}",
        "bars": len(df),
        "freq": freq,
        "last_close": last_close,
        "atr": float(atr) if pd.notna(atr) else 0.0,
        "lz": {
            "upper_bound": lz_upper,
            "lower_bound": lz_lower,
            "active_upper_zones": len([z for z in active_lz_zones if z["kind"] == "upper"]),
            "active_lower_zones": len([z for z in active_lz_zones if z["kind"] == "lower"]),
            "nearest_upper_zones": upper_zones_info[:3],
            "nearest_lower_zones": lower_zones_info[:3],
        },
        "bb": {
            "period": cfg.bb_period,
            "std_mult": cfg.bb_std,
            "upper": bb_upper_last,
            "mid": float(bb_mid.iloc[-1]) if pd.notna(bb_mid.iloc[-1]) else None,
            "lower": bb_lower_last,
        },
    }

    return SpaceOutput(
        lz_space=lz_space,
        bb_space=bb_space,
        lz_zones=lz_zones,
        debug=debug_info,
    )


def build_space_html(
    df: pd.DataFrame,
    output: SpaceOutput,
    cfg: SpaceConfig,
    out_path: str,
    zones: List[Dict],
    offset: int = 0,
) -> None:
    """生成HTML可视化报告"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_str = [str(idx) for idx in df.index]

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

    shapes = []
    for z in zones:
        pivot_i = int(z["pivot_i"]) - offset
        if pivot_i < 0 or pivot_i >= len(df):
            continue

        x_left = x_str[pivot_i]
        x_right_i = min(pivot_i + 10, len(df) - 1)
        x_right = x_str[x_right_i]

        if z["kind"] == "upper":
            y0 = z["base"]
            y1 = z["y"]
        else:
            y0 = z["y"]
            y1 = z["base"]

        line_style = "dash" if z["grabbed"] else "solid"
        line_width = 1 if z["grabbed"] else 2

        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=x_left, x1=x_right,
            y0=y0, y1=y1,
            line=dict(color=z["color"], width=1),
            fillcolor=z["color"],
            layer="below",
            opacity=0.3,
        ))

        x2_i = int(z["x2"]) - offset
        x2_i = max(pivot_i, min(x2_i, len(df) - 1))
        fig.add_trace(
            go.Scatter(
                x=[x_str[pivot_i], x_str[x2_i]],
                y=[z["y"], z["y"]],
                mode="lines",
                line=dict(color=z["color"], width=line_width, dash=line_style),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1,
        )

    bb_upper, bb_mid, bb_lower = _calc_bollinger_bands(df, cfg.bb_period, cfg.bb_std)

    fig.add_trace(
        go.Scatter(
            x=x_str,
            y=bb_upper,
            mode="lines",
            line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash"),
            name="BB Upper",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_str,
            y=bb_lower,
            mode="lines",
            line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash"),
            name="BB Lower",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_str,
            y=bb_mid,
            mode="lines",
            line=dict(color="rgba(255,193,7,0.4)", width=1),
            name="BB Mid",
            showlegend=False,
        ),
        row=1, col=1,
    )

    if output.lz_space.upper_bound:
        fig.add_hline(
            y=output.lz_space.upper_bound,
            line=dict(color="#2370a3", width=2, dash="dot"),
            annotation=dict(text=f"LZ上界 {output.lz_space.upper_bound:.2f}", font=dict(size=10)),
            row=1,
        )
    if output.lz_space.lower_bound:
        fig.add_hline(
            y=output.lz_space.lower_bound,
            line=dict(color="#23a372", width=2, dash="dot"),
            annotation=dict(text=f"LZ下界 {output.lz_space.lower_bound:.2f}", font=dict(size=10)),
            row=1,
        )
    if output.lz_space.stop_price:
        fig.add_hline(
            y=output.lz_space.stop_price,
            line=dict(color="#ef5350", width=1, dash="dash"),
            annotation=dict(text=f"止损 {output.lz_space.stop_price:.2f}", font=dict(size=9)),
            row=1,
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

    lz_stop_str = f"{output.lz_space.stop_price:.2f}" if output.lz_space.stop_price else "N/A"
    bb_stop_str = f"{output.bb_space.stop_price:.2f}" if output.bb_space.stop_price else "N/A"

    info_html = (
        f"<b>Space Script 输出</b><br>"
        f"<br><b>Liquidity Zones 空间</b><br>"
        f"上界: {output.lz_space.upper_bound:.2f}<br>"
        f"下界: {output.lz_space.lower_bound:.2f}<br>"
        f"止损: {lz_stop_str}<br>"
        f"盈亏比: {output.lz_space.rr:.2f} ({'✓' if output.lz_space.rr_pass else '✗'})<br>"
        f"剩余空间: {output.lz_space.space_left:.2f} ({output.lz_space.space_left_atr:.2f} ATR)<br>"
        f"<br><b>Bollinger Bands 空间</b><br>"
        f"上界: {output.bb_space.upper_bound:.2f}<br>"
        f"下界: {output.bb_space.lower_bound:.2f}<br>"
        f"止损: {bb_stop_str}<br>"
        f"盈亏比: {output.bb_space.rr:.2f} ({'✓' if output.bb_space.rr_pass else '✗'})<br>"
        f"剩余空间: {output.bb_space.space_left:.2f} ({output.bb_space.space_left_atr:.2f} ATR)<br>"
        f"<br><b>通用信息</b><br>"
        f"最新价: {output.debug['last_close']:.2f}<br>"
        f"ATR: {output.debug['atr']:.2f}<br>"
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
        font=dict(size=11, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=8,
    )

    fig.update_layout(
        title=f"Space Script - {output.debug['symbol']} ({cfg.tf_mid})",
        shapes=shapes,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=900,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False, type="category")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="000426", help="股票代码")
    ap.add_argument("--freq", type=str, default="60", help="周期: 60(60分钟)")
    ap.add_argument("--years", type=float, default=1, help="数据年限")
    ap.add_argument("--out", type=str, default="", help="输出HTML路径(可选)")

    ap.add_argument("--lz-left-bars", type=int, default=10, help="LZ leftBars")
    ap.add_argument("--lz-qty-pivots", type=int, default=10, help="LZ zones数量")
    ap.add_argument("--lz-flt", type=str, default="Mid", choices=["Low", "Mid", "High"], help="LZ过滤强度")
    ap.add_argument("--lz-dynamic", action="store_true", help="LZ动态距离")
    ap.add_argument("--bb-period", type=int, default=20, help="布林带周期")
    ap.add_argument("--bb-std", type=float, default=2.0, help="布林带标准差倍数")
    ap.add_argument("--sl-buffer-atr", type=float, default=0.5, help="止损ATR缓冲")
    ap.add_argument("--rr-min", type=float, default=1.5, help="最小盈亏比")
    ap.add_argument("--space-th", type=float, default=1.0, help="止盈空间阈值(ATR)")

    args = ap.parse_args()

    end = datetime.now().date()
    freq_arg = args.freq.strip().lower()
    if freq_arg in ["d", "day", "daily"]:
        freq = "d"
        calc_days = 1600
    else:
        try:
            _ = int(freq_arg)
        except ValueError as e:
            raise ValueError("--freq 只能是 d 或 1/5/15/30/60（分钟，数字）") from e
        freq = freq_arg
        calc_days = 1200

    start = end - timedelta(days=calc_days)

    cfg = SpaceConfig(
        tf_mid=args.freq,
        lz_left_bars=args.lz_left_bars,
        lz_qty_pivots=args.lz_qty_pivots,
        lz_flt=args.lz_flt,
        lz_dynamic=args.lz_dynamic,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        sl_buffer_atr=args.sl_buffer_atr,
        rr_min=args.rr_min,
        space_th=args.space_th,
    )

    print(f"\n{'='*60}")
    print(f"Space Script (B脚本) - 空间判定")
    print(f"{'='*60}")
    print(f"股票代码: {args.symbol}")
    print(f"时间范围: {start} ~ {end}")
    print(f"周期: {args.freq}")
    print(f"{'='*60}\n")

    output = space_script(args.symbol, str(start), str(end), cfg)

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
        df_full = fetch_kline_pytdx(args.symbol, str(start), str(end), cfg.tf_mid)
        df_show = df_full.tail(500)
        
        idx0 = df_show.index[0]
        idx1 = df_show.index[-1]
        offset = len(df_full) - len(df_show)
        
        zones_kept = []
        for z in output.lz_zones:
            pt = z["pivot_time"]
            if (pt >= idx0) and (pt <= idx1):
                zones_kept.append(z)
            else:
                x2_i = int(z["x2"]) - offset
                if x2_i >= 0:
                    zones_kept.append(z)
        
        build_space_html(df_show, output, cfg, args.out, zones_kept, offset)


if __name__ == "__main__":
    main()
