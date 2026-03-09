# -*- coding: utf-8 -*-
"""
AMP + Divergence(Many) + Liquidity Zones -> One Plotly HTML (pytdx)

数据源：直接复用你附件里的 pytdx_client.py（connect_pytdx/get_kline_data）

Usage example:
    python three_indicators_plotly.py --symbol 600547 --freq d --bars 800 --out three.html --div-showlines --lz-show-pivots
    python three_indicators_plotly.py --symbol 000426 --freq 60m --bars 1200 --out three_60m.html

Notes:
- 该脚本把 3 个指标叠加到同一张 K 线图上：
  - AMP：通道/活跃线/剖面 + 信息框
  - Liquidity Zones：箱体 + 水平线 + grabbed 标记
  - Divergence(Many)：背离连线 + 标签
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_module(name: str, path: str):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MOD_PYT = _load_module("pytdx_client", str(ROOT / "datasource" / "pytdx_client.py"))
MOD_AMP = _load_module("amp_plotly", str(HERE / "amp_plotly.py"))
MOD_DIV = _load_module("divergence_many_plotly", str(HERE / "divergence_many_plotly.py"))
MOD_LZ = _load_module("liquidity_zones_plotly", str(HERE / "liquidity_zones_plotly.py"))


def fetch_df(symbol: str, freq: str, bars: int) -> Tuple[pd.DataFrame, str]:
    api = MOD_PYT.connect_pytdx()
    try:
        raw = MOD_PYT.get_kline_data(api, symbol, freq, int(bars))
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"pytdx 无数据：symbol={symbol} freq={freq} bars={bars}")

        df = raw.copy()
        if "datetime" not in df.columns:
            raise RuntimeError("get_kline_data 返回缺少 datetime 列")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")

        need_cols = ["open", "high", "low", "close", "volume"]
        for c in need_cols:
            if c not in df.columns:
                raise RuntimeError(f"数据缺少列：{c}（现有列={list(df.columns)}）")
        df = df[need_cols].copy()
        for c in need_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).copy()
        df["volume"] = df["volume"].fillna(0.0)

        stock_name = ""
        if hasattr(MOD_PYT, "get_stock_name"):
            try:
                stock_name = MOD_PYT.get_stock_name(api, symbol) or ""
            except Exception:
                stock_name = ""

        return df, stock_name
    finally:
        api.disconnect()


def build_base_fig(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.78, 0.22],
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
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
        row=1,
        col=1,
    )

    vol_colors = np.where(df["close"] >= df["open"], "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], marker_color=vol_colors, showlegend=False),
        row=2,
        col=1,
    )

    fig.update_layout(
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=980,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")
    return fig


def overlay_amp(fig: go.Figure, df: pd.DataFrame, symbol: str, stock_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = MOD_AMP.AMPConfig(
        useAdaptive=(not args.no_adaptive),
        pI=int(args.amp_period),
        devMultiplier=float(args.amp_dev),
        uL=bool(args.amp_use_log),
        activityMethod=("Touches" if args.amp_touches else "Volume"),
        nFills=int(args.amp_fills),
        numActivityLines=int(args.amp_active_lines),
        showProfile=(not args.amp_no_profile),
        showRegLine=bool(args.amp_show_mid),
    )
    payload = MOD_AMP.compute_amp_last(df, cfg)
    m = payload["metrics"]
    ch = payload["channel"]

    fig.add_trace(
        go.Scatter(
            x=ch["x"],
            y=ch["upper"],
            mode="lines",
            line=dict(color=cfg.regColor, width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ch["x"],
            y=ch["lower"],
            mode="lines",
            line=dict(color=cfg.regColor, width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ch["x"],
            y=ch["upper"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ch["x"],
            y=ch["lower"],
            fill="tonexty",
            fillcolor=cfg.channelFill,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if ch.get("showMid"):
        fig.add_trace(
            go.Scatter(
                x=ch["x"],
                y=ch["mid"],
                mode="lines",
                line=dict(color="rgba(180,180,180,0.7)", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    for al in payload.get("activityLines", []):
        fig.add_trace(
            go.Scatter(
                x=al["x"],
                y=al["y"],
                mode="lines",
                line=dict(color=al["color"], width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    for poly in payload.get("profilePolys", []):
        x0 = poly["x0"]
        x1 = poly["x1"]
        y0t = poly["y0_top"]
        y0b = poly["y0_bot"]
        y1t = poly["y1_top"]
        y1b = poly["y1_bot"]
        shapes.append(
            dict(
                type="path",
                xref="x",
                yref="y",
                path=f"M {x0} {y0t} L {x0} {y0b} L {x1} {y1b} L {x1} {y1t} Z",
                fillcolor=poly["fill"],
                line=dict(width=0),
                layer="below",
                opacity=1.0,
            )
        )
    fig.update_layout(shapes=shapes)

    ul = m["upper_slope"]
    ll = m["lower_slope"]
    pearson_best = m.get("pearson_best_from_devATF", None)
    pearson_best_str = "-" if pearson_best is None else f"{pearson_best:.3f}"

    lines_info = ""
    for st in m.get("active_lines_stats", []):
        lines_info += (
            f"线{st['rank']}: 起点横向位置={st['x_start_pos_0_1']:.3f}, "
            f"起点高度占比={st['height_start_0_1']:.3f}, "
            f"终点高度占比={st['height_end_0_1']:.3f}<br>"
        )
    if not lines_info:
        lines_info = "(无活跃线)<br>"

    stock_display = stock_name if stock_name else symbol
    info_html = (
        f"<b>AMP</b> ({stock_display})<br>"
        f"窗口（bar）：{m['window_len']} | finalPeriod：{m['finalPeriod']}<br>"
        f"强度 pR：{m['strength_pR_cD']:.3f} | DevATF Pearson：{pearson_best_str}<br>"
        f"收盘：{m['close_last']:.2f} | 位置(0~1)：{m['close_pos_0_1']:.3f}<br>"
        f"上轨：{m['upper_start']:.2f} → {m['upper_end']:.2f}<br>"
        f"下轨：{m['lower_start']:.2f} → {m['lower_end']:.2f}<br>"
        f"<br><b>斜率/收益</b><br>"
        f"上轨 每bar收益：{ul['ret_per_bar']:.5f} | 窗口总收益：{ul['total_ret']:.3f}<br>"
        f"下轨 每bar收益：{ll['ret_per_bar']:.5f} | 窗口总收益：{ll['total_ret']:.3f}<br>"
        f"<br><b>Most Active Lines</b><br>{lines_info}"
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
        x=ch["x"][0],
        y=m["lower_start"],
        xref="x",
        yref="y",
        text=f"{m['strength_pR_cD']:.3f}",
        showarrow=False,
        font=dict(size=12, color="rgba(200,200,200,0.9)"),
        bgcolor="rgba(0,0,0,0.25)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=4,
        yshift=20,
    )

    return payload


def overlay_liquidity_zones(fig: go.Figure, df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = MOD_LZ.LZConfig(
        leftBars=int(args.lz_leftBars),
        qty_pivots=int(args.lz_qty_pivots),
        flt=str(args.lz_flt),
        dynamic=bool(args.lz_dynamic),
        hidePivot=bool(args.lz_hidePivot),
    )
    payload = MOD_LZ.build_liquidity_zones(df, cfg)

    zones = payload["zones"]
    pivot_pts = payload["pivot_points"]

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []

    grabbed_x, grabbed_y = [], []

    for z in zones:
        pivot_i = int(z["pivot_i"])
        if pivot_i < 0 or pivot_i >= len(df):
            continue
        x_left = df.index[pivot_i]
        x_right_i = min(pivot_i + cfg.box_width_bars, len(df) - 1)
        x_right = df.index[x_right_i]

        if z["kind"] == "upper":
            y0, y1 = z["base"], z["y"]
        else:
            y0, y1 = z["y"], z["base"]

        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x_left,
                x1=x_right,
                y0=y0,
                y1=y1,
                line=dict(color=z["color"], width=2 if not z["grabbed"] else 1),
                fillcolor=z["color"] if not z["grabbed"] else "rgba(0,0,0,0)",
                layer="below",
            )
        )

        x_mid = df.index[min(pivot_i + cfg.box_width_bars // 2, len(df) - 1)]
        y_mid = (y0 + y1) / 2.0
        annotations.append(
            dict(
                x=x_mid,
                y=y_mid,
                xref="x",
                yref="y",
                text=str(z["text"]).replace("\n", "<br>"),
                showarrow=False,
                font=dict(color="#c9d1d9", size=10),
                bgcolor="rgba(0,0,0,0.0)",
                align="left",
            )
        )

        x2_i = max(pivot_i, min(int(z["x2"]), len(df) - 1))
        xs = [df.index[pivot_i], df.index[x2_i]]
        ys = [z["y"], z["y"]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=z["color"], width=int(z["line_width"]), dash=z["line_style"]),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        if z["grabbed"] and (z["grabbed_i"] is not None):
            gi = int(z["grabbed_i"])
            if 0 <= gi < len(df):
                grabbed_x.append(df.index[gi])
                grabbed_y.append(z["y"])

    if grabbed_x:
        fig.add_trace(
            go.Scatter(
                x=grabbed_x,
                y=grabbed_y,
                mode="text",
                text=["〇"] * len(grabbed_x),
                textfont=dict(color="#df1c1c", size=16),
                hovertemplate="Liquidity Grabbed<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if args.lz_show_pivots:
        xs_h, ys_h, xs_l, ys_l = [], [], [], []
        for p in pivot_pts:
            if p.get("kind") == "ph":
                xs_h.append(p["time"])
                ys_h.append(p["y"])
            else:
                xs_l.append(p["time"])
                ys_l.append(p["y"])
        if xs_h:
            fig.add_trace(
                go.Scatter(
                    x=xs_h,
                    y=ys_h,
                    mode="markers+text",
                    text=["H"] * len(xs_h),
                    textposition="top center",
                    marker=dict(size=6),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        if xs_l:
            fig.add_trace(
                go.Scatter(
                    x=xs_l,
                    y=ys_l,
                    mode="markers+text",
                    text=["L"] * len(xs_l),
                    textposition="bottom center",
                    marker=dict(size=6),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    fig.update_layout(shapes=shapes, annotations=annotations)
    return payload


def overlay_divergence(fig: go.Figure, df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = MOD_DIV.DivConfig(
        prd=int(args.div_prd),
        source=str(args.div_source),
        searchdiv=str(args.div_searchdiv),
        showindis=str(args.div_showindis),
        showlimit=int(args.div_showlimit),
        maxpp=int(args.div_maxpp),
        maxbars=int(args.div_maxbars),
        shownum=bool(args.div_shownum),
        showlast=bool(args.div_showlast),
        dontconfirm=bool(args.div_dontconfirm),
        showlines=bool(args.div_showlines),
        showpivot=bool(args.div_showpivot),
    )

    for k in [
        "calcmacd",
        "calcmacda",
        "calcrsi",
        "calcstoc",
        "calccci",
        "calcmom",
        "calcobv",
        "calcvwmacd",
        "calccmf",
        "calcmfi",
        "calcext",
    ]:
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, bool(v))

    pos_lines, neg_lines, pos_labels, neg_labels, last_feat = MOD_DIV.run_divergence_engine(df, cfg)

    xcat = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M") for x in df.index]
    cat2ts = {c: ts for c, ts in zip(xcat, df.index)}

    def _map_x_list(x_list: List[Any]) -> List[Any]:
        out = []
        for x in x_list:
            if isinstance(x, str) and x in cat2ts:
                out.append(cat2ts[x])
            else:
                out.append(x)
        return out

    def add_lines(lines: List[Dict[str, Any]]):
        for ln in lines:
            fig.add_trace(
                go.Scatter(
                    x=_map_x_list(list(ln["x"])),
                    y=list(ln["y"]),
                    mode="lines",
                    line=dict(width=int(ln["width"]), dash=str(ln["dash"])),
                    marker=dict(color=str(ln["color"])),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    if cfg.showlines:
        add_lines(pos_lines)
        add_lines(neg_lines)

    for (x, y, text_, bg, tc) in neg_labels:
        fig.add_annotation(
            x=cat2ts.get(x, x),
            y=y,
            xref="x",
            yref="y",
            text=str(text_).replace("\n", "<br>"),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-25,
            bgcolor=bg,
            font=dict(color=tc, size=12),
            bordercolor=bg,
            borderwidth=1,
        )

    for (x, y, text_, bg, tc) in pos_labels:
        fig.add_annotation(
            x=cat2ts.get(x, x),
            y=y,
            xref="x",
            yref="y",
            text=str(text_).replace("\n", "<br>"),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=25,
            bgcolor=bg,
            font=dict(color=tc, size=12),
            bordercolor=bg,
            borderwidth=1,
        )

    return {"last_feat": last_feat, "cfg": cfg}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--symbol", type=str, default="000426", help="A股代码，如 000426/600547")
    ap.add_argument(
        "--freq",
        type=str,
        default="d",
        help="周期：1m,5m,15m,30m,60m,d,w,m（与 pytdx_client.PERIOD_MAP 对齐）",
    )
    ap.add_argument(
        "--bars",
        type=int,
        default=800,
        help="获取/展示最近 N 根K线（建议>=700用于LZ的stdev/ATR热身）",
    )
    ap.add_argument("--out", type=str, default="three_indicators.html", help="输出 HTML 文件名")
    ap.add_argument("--png", type=str, default="", help="可选：导出 PNG（需 kaleido）")

    # AMP args
    ap.add_argument("--amp-use-log", action="store_true", dest="amp_use_log")
    ap.add_argument("--no-adaptive", action="store_true", help="AMP关闭自适应")
    ap.add_argument("--amp-period", type=int, default=200)
    ap.add_argument("--amp-dev", type=float, default=2.0)
    ap.add_argument("--amp-fills", type=int, default=23)
    ap.add_argument("--amp-active-lines", type=int, default=2)
    ap.add_argument("--amp-touches", action="store_true")
    ap.add_argument("--amp-no-profile", action="store_true")
    ap.add_argument("--amp-show-mid", action="store_true")

    # Liquidity Zones args
    ap.add_argument("--lz-leftBars", type=int, default=10)
    ap.add_argument("--lz-qty-pivots", type=int, default=10)
    ap.add_argument("--lz-flt", type=str, default="Mid", choices=["Low", "Mid", "High"])
    ap.add_argument("--lz-dynamic", action="store_true")
    ap.add_argument("--lz-hidePivot", action="store_true")
    ap.add_argument("--lz-show-pivots", action="store_true", help="在图上显示过滤后的pivot点(H/L)")

    # Divergence args
    ap.add_argument("--div-prd", type=int, default=5)
    ap.add_argument("--div-source", type=str, default="Close", choices=["Close", "High/Low"])
    ap.add_argument("--div-searchdiv", type=str, default="Regular", choices=["Regular", "Hidden", "Regular/Hidden"])
    ap.add_argument("--div-showindis", type=str, default="Full", choices=["Full", "First Letter", "Don't Show"])
    ap.add_argument("--div-showlimit", type=int, default=1)
    ap.add_argument("--div-maxpp", type=int, default=10)
    ap.add_argument("--div-maxbars", type=int, default=100)
    ap.add_argument("--div-shownum", action="store_true")
    ap.add_argument("--div-showlast", action="store_true")
    ap.add_argument("--div-dontconfirm", action="store_true")
    ap.add_argument("--div-showlines", action="store_true", help="显示背离连线")
    ap.add_argument("--div-showpivot", action="store_true")

    # optional enable/disable individual indicators for divergence
    for k in [
        "calcmacd",
        "calcmacda",
        "calcrsi",
        "calcstoc",
        "calccci",
        "calcmom",
        "calcobv",
        "calcvwmacd",
        "calccmf",
        "calcmfi",
        "calcext",
    ]:
        ap.add_argument(f"--{k}", action="store_true", help=f"divergence: enable {k}")
        ap.add_argument(f"--no-{k}", action="store_false", dest=k, help=f"divergence: disable {k}")
        ap.set_defaults(**{k: None})

    args = ap.parse_args()

    df, stock_name = fetch_df(args.symbol, args.freq, args.bars)
    if len(df) < 60:
        raise RuntimeError("数据太少（<60根），无法计算 AMP")

    fig = build_base_fig(df)

    overlay_liquidity_zones(fig, df, args)
    overlay_divergence(fig, df, args)
    overlay_amp(fig, df, args.symbol, stock_name, args)

    stock_display = stock_name if stock_name else args.symbol
    fig.update_layout(title=f"{stock_display} | {args.freq} | bars={len(df)} | AMP+DIV+LZ")

    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {args.out}")

    if args.png:
        fig.write_image(args.png, scale=2)
        print(f"[OK] PNG saved: {args.png}")


if __name__ == "__main__":
    main()
