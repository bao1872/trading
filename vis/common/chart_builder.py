# -*- coding: utf-8 -*-
"""
Purpose:
    通用图表组件，供交易工作台各页面共享。
    净值曲线、回撤曲线、持仓分布等 Plotly 图表。

Inputs:
    - DataFrame（由 data_loader 提供）

Outputs:
    - plotly.graph_objects.Figure 对象

How to Run:
    python -m vis.common.chart_builder

Examples:
    from vis.common.chart_builder import build_equity_curve_chart
    fig = build_equity_curve_chart(eq_df)

Side Effects:
    - 无（纯图表构建，不读写文件）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vis.common.theme import DARK_THEME


def build_equity_curve_chart(eq_df: pd.DataFrame, period: str = "all") -> go.Figure:
    """构建净值曲线 + 回撤曲线双轴图

    Args:
        eq_df: 净值曲线 DataFrame（需含 date, nav_live, drawdown 列）
        period: 区间选择 "20d" / "60d" / "all"
    """
    if eq_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="净值曲线（无数据）",
            template="plotly_dark",
            paper_bgcolor=DARK_THEME["paper_bgcolor"],
            plot_bgcolor=DARK_THEME["bg_color"],
            font_color=DARK_THEME["text_color"],
        )
        return fig

    df = eq_df.copy()
    if "date" in df.columns:
        df = df.sort_values("date")

    if period == "20d" and len(df) > 20:
        df = df.tail(20)
    elif period == "60d" and len(df) > 60:
        df = df.tail(60)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"] if "date" in df.columns else df.index,
            y=df["nav_live"],
            name="NAV",
            line=dict(color=DARK_THEME["up_color"], width=2),
            hovertemplate="日期: %{x}<br>净值: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    if "drawdown" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"] if "date" in df.columns else df.index,
                y=df["drawdown"] * 100 if df["drawdown"].max() > -1 else df["drawdown"],
                name="回撤",
                fill="tozeroy",
                line=dict(color=DARK_THEME["down_color"], width=1),
                fillcolor="rgba(239,83,80,0.15)",
                hovertemplate="日期: %{x}<br>回撤: %{y:.2f}%<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["bg_color"],
        font_color=DARK_THEME["text_color"],
        height=400,
        margin=dict(l=60, r=20, t=30, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridcolor=DARK_THEME["grid_color"], row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor=DARK_THEME["grid_color"], row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=DARK_THEME["grid_color"], row=1, col=1, title_text="净值")
    fig.update_yaxes(showgrid=True, gridcolor=DARK_THEME["grid_color"], row=2, col=1, title_text="回撤%")

    return fig


def build_daily_return_bar(eq_df: pd.DataFrame, n_days: int = 30) -> go.Figure:
    """构建日收益率柱状图"""
    if eq_df.empty or "daily_return" not in eq_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="日收益率（无数据）",
            template="plotly_dark",
            paper_bgcolor=DARK_THEME["paper_bgcolor"],
            font_color=DARK_THEME["text_color"],
        )
        return fig

    df = eq_df.copy().sort_values("date").tail(n_days)
    colors = [
        DARK_THEME["up_color"] if r >= 0 else DARK_THEME["down_color"]
        for r in df["daily_return"]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["daily_return"] * 100 if df["daily_return"].abs().max() < 1 else df["daily_return"],
            marker_color=colors,
            hovertemplate="日期: %{x}<br>收益率: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["bg_color"],
        font_color=DARK_THEME["text_color"],
        height=250,
        margin=dict(l=60, r=20, t=30, b=30),
        title="日收益率",
        yaxis_title="收益率%",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=DARK_THEME["grid_color"])
    fig.update_yaxes(showgrid=True, gridcolor=DARK_THEME["grid_color"])

    return fig


def build_holdings_pie(holdings_df: pd.DataFrame, name_map: dict = None) -> go.Figure:
    """构建持仓分布饼图"""
    if holdings_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="持仓分布（无数据）",
            template="plotly_dark",
            paper_bgcolor=DARK_THEME["paper_bgcolor"],
            font_color=DARK_THEME["text_color"],
        )
        return fig

    labels = []
    for _, row in holdings_df.iterrows():
        ts = row.get("ts_code", row.get("code", ""))
        name = (name_map or {}).get(ts, ts.split(".")[0] if "." in ts else ts)
        labels.append(name)

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=holdings_df["weight"] * 100 if "weight" in holdings_df.columns else None,
            hole=0.4,
            textinfo="label+percent",
            marker=dict(
                line=dict(color=DARK_THEME["bg_color"], width=2),
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        font_color=DARK_THEME["text_color"],
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        title="持仓分布",
        showlegend=False,
    )

    return fig


if __name__ == "__main__":
    from vis.common.data_loader import load_equity_curve, load_holdings_df, load_latest_date

    print("=" * 60)
    print("  chart_builder 自测")
    print("=" * 60)

    eq = load_equity_curve()
    if not eq.empty:
        fig1 = build_equity_curve_chart(eq)
        print(f"  ✅ 净值曲线图: {len(eq)} 个数据点")

        fig2 = build_daily_return_bar(eq)
        print(f"  ✅ 日收益率图")
    else:
        print("  ⚠️ 无净值曲线数据，跳过图表测试")

    latest = load_latest_date()
    if latest:
        holdings = load_holdings_df(latest)
        if not holdings.empty:
            fig3 = build_holdings_pie(holdings)
            print(f"  ✅ 持仓分布图: {len(holdings)} 只")
        else:
            print("  ⚠️ 无持仓数据")

    print(f"\n  ✅ chart_builder 自测完成")
