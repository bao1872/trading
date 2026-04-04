# -*- coding: utf-8 -*-
"""
MFE/MAE 标签生成与可视化（5m, H=3天）- Plotly HTML

目标：
- 为 5m 执行周期生成两条“未来窗口极值”标签（做多视角）：
  1) MFE_long: 未来H根内最高价相对当前close的最大上行幅度（ATR标准化）
  2) MAE_long: 未来H根内最低价相对当前close的最大下行幅度（ATR标准化，通常为负）

输出：
- HTML：主图K线 + 2个副图（MFE_long 与 MAE_long）
- CSV：可选导出包含标签的表

说明（避免时间错位）：
- 标签窗口使用 t+1 ... t+H（不包含当前bar），最后H根会是 NaN

用法示例：
    python labels_mfe_mae_h3_5m_plotly.py --symbol 600547 --days 40 --out sdhj_mfe_mae_h3_5m.html
    python labels_mfe_mae_h3_5m_plotly.py --symbol 600547 --days 40 --out sdhj_mfe_mae_h3_5m.html --csv sdhj_mfe_mae_h3_5m.csv

依赖：
    pip install pandas numpy plotly pytdx
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP

# =========================
# Data source (pytdx)
# =========================

def _category_from_freq(freq: str) -> int:
    f = str(freq).strip().lower()
    if f in ("5", "5m"):
        return 0
    if f in ("1", "1m"):
        return 7
    if f in ("15", "15m"):
        return 1
    if f in ("30", "30m"):
        return 2
    if f in ("60", "60m", "1h"):
        return 3
    if f in ("d", "day", "daily", "101"):
        return 4
    raise ValueError(f"不支持的频率: {freq}")


def _connect_pytdx(timeout: int = 2) -> TdxHq_API:
    servers = [
        ("119.147.212.81", 7709),
        ("119.147.164.60", 7709),
        ("14.215.128.18", 7709),
        ("14.215.128.116", 7709),
        ("101.133.156.38", 7709),
        ("114.80.149.19", 7709),
        ("115.238.90.165", 7709),
        ("123.125.108.23", 7709),
        ("180.153.18.170", 7709),
        ("202.108.253.131", 7709),
    ]
    last_errors = []
    for host, port in servers:
        try:
            api = TdxHq_API(raise_exception=True, auto_retry=True)
            if api.connect(host, port, time_out=timeout):
                return api
        except TdxConnectionError as exc:
            last_errors.append(f"{host}:{port} {exc}")
        except Exception as exc:
            last_errors.append(f"{host}:{port} {exc}")
    err = "; ".join(last_errors[-5:])
    raise RuntimeError(f"pytdx连接失败; {err}")


def fetch_kline_pytdx(
    symbol: str,
    start: str,
    end: str,
    freq: str = "5m",
    api: TdxHq_API | None = None,
) -> pd.DataFrame:
    """
    拉取 pytdx K线并按时间过滤。
    注意：pytdx bars 接口是“从最新往回翻页”，因此需要循环翻页后再过滤时间范围。
    """
    own_api = api is None
    if api is None:
        api = connect_pytdx()
    try:
        cat = _category_from_freq(freq)
        mkt = 1 if symbol.startswith("6") else 0

        page = 0
        size = 700
        frames = []

        while True:
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
            if not recs:
                break
            d = pd.DataFrame(recs)

            if "datetime" in d.columns:
                d["dt"] = pd.to_datetime(d["datetime"]).tz_localize(None)
            elif {"year", "month", "day", "hour", "minute"}.issubset(d.columns):
                d["dt"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]].astype(int))
            else:
                raise RuntimeError("pytdx返回数据缺少时间列")

            if "vol" in d.columns:
                d = d.rename(columns={"vol": "volume"})
            cols = ["dt", "open", "high", "low", "close"] + (["volume"] if "volume" in d.columns else [])
            d = d[cols].sort_values("dt")
            frames.append(d)

            if len(recs) < size:
                break
            page += 1

        if not frames:
            raise RuntimeError("pytdx无数据")

        df = pd.concat(frames, ignore_index=True).sort_values("dt")
        df = df.drop_duplicates(subset=["dt"], keep="last").set_index("dt")

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        return df
    finally:
        if own_api:
            api.disconnect()


# =========================
# Indicators / Labels
# =========================

def atr_wilder(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Wilder ATR（EMA alpha=1/n），更接近交易软件口径。
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    return atr


def compute_mfe_mae_long_labels(df: pd.DataFrame, horizon_bars: int, atr_n: int = 20) -> pd.DataFrame:
    """
    计算做多视角标签：
      future_high(t) = max(high[t+1 : t+h])
      future_low(t)  = min(low[t+1 : t+h])

      MFE_long = (future_high - close) / ATR
      MAE_long = (future_low  - close) / ATR   # 通常为负
    """
    out = df.copy()
    out["ATR"] = atr_wilder(out, n=atr_n)

    h = int(horizon_bars)
    if h <= 0:
        raise ValueError("horizon_bars 必须 > 0")

    # t+1..t+h：先 shift(-1) 再 rolling(h) 计算窗口极值，最后 shift(-(h-1)) 对齐回 t
    s_high = out["high"].shift(-1)
    s_low = out["low"].shift(-1)

    future_high = s_high.rolling(window=h, min_periods=h).max().shift(-(h - 1))
    future_low = s_low.rolling(window=h, min_periods=h).min().shift(-(h - 1))

    out["future_high"] = future_high
    out["future_low"] = future_low

    atr = out["ATR"].replace(0.0, np.nan)
    out["MFE_long"] = (out["future_high"] - out["close"]) / atr
    out["MAE_long"] = (out["future_low"] - out["close"]) / atr

    # 常用的风险幅度（非负）
    out["RISK_long"] = (out["close"] - out["future_low"]) / atr

    return out


# =========================
# Plot
# =========================

def plot_labels(df: pd.DataFrame, title: str, out_html: str) -> None:
    # 用 category 轴避免非交易时段空档
    x = df.index.strftime("%Y-%m-%d %H:%M")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.22, 0.18],
        subplot_titles=("Price (5m)", "MFE_long & MAE_long (ATR units)", "SCORE_buy = MFE - λ * RISK (ATR units)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # MFE & MAE in the same subplot
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["MFE_long"],
            mode="lines",
            name="MFE_long",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["MAE_long"],
            mode="lines",
            name="MAE_long",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, row=2, col=1)

    # SCORE subplot
    if "SCORE_buy" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["SCORE_buy"],
                mode="lines",
                name="SCORE_buy",
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0, row=3, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=40),
    )

    # category axis
    fig.update_xaxes(type="category", showgrid=False)

    # 更易读
    fig.update_yaxes(showgrid=True, zeroline=False)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] saved -> {out_html}")


# =========================
# CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MFE/MAE long labels (5m, H=3 days) and plot to HTML.")
    parser.add_argument("--symbol", required=True, help="A股代码，如 600547")
    parser.add_argument("--freq", default="5m", help="频率（默认5m）")
    parser.add_argument("--days", type=int, default=60, help="向前回看多少个自然日（建议>=40）")
    parser.add_argument("--hdays", type=int, default=3, help="标签窗口：未来多少个交易日（默认3天）")
    parser.add_argument("--bars_per_day", type=int, default=48, help="每个交易日的bar数（5m默认48）")
    parser.add_argument("--atr", type=int, default=20, help="ATR窗口（Wilder smoothing）")
    parser.add_argument("--lam", type=float, default=1.0, help="score = MFE - lam * risk，risk = -MAE（默认1.0）")
    parser.add_argument("--out", required=True, help="输出HTML文件名，如 out.html")
    parser.add_argument("--csv", default="", help="可选：导出CSV（包含标签列）")

    args = parser.parse_args()

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(args.days))
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    df = fetch_kline_pytdx(args.symbol, start=start, end=end, freq=args.freq)
    if df.empty:
        raise RuntimeError("拉取K线为空，请扩大 --days 或检查代码/网络。")

    horizon_bars = int(args.hdays) * int(args.bars_per_day)
    lab = compute_mfe_mae_long_labels(df, horizon_bars=horizon_bars, atr_n=int(args.atr))
    # 综合评分：机会-风险（risk = -MAE = RISK_long）
    lam = float(args.lam)
    lab["SCORE_buy"] = lab["MFE_long"] - lam * lab["RISK_long"]

    title = f"{args.symbol} | {args.freq} | H={args.hdays}d ({horizon_bars} bars) | Labels: MFE_long / MAE_long | λ={args.lam}"
    plot_labels(lab, title=title, out_html=args.out)

    if args.csv:
        out_csv = args.csv
        lab_out = lab.copy()
        lab_out.index.name = "dt"
        lab_out.to_csv(out_csv, encoding="utf-8-sig")
        print(f"[OK] saved -> {out_csv}")

    # quick peek
    last = lab.dropna(subset=["MFE_long", "MAE_long"]).tail(1)
    if not last.empty:
        r = last.iloc[0]
        print(
            "[LAST VALID] dt=%s close=%.3f ATR=%.3f MFE=%.3f MAE=%.3f (risk=%.3f score=%.3f)"
            % (str(last.index[0]), float(r["close"]), float(r["ATR"]), float(r["MFE_long"]), float(r["MAE_long"]), float(r["RISK_long"]), float(r.get("SCORE_buy", np.nan)))
        )


if __name__ == "__main__":
    main()
