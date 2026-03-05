# -*- coding: utf-8 -*-
"""
行情快照聚合脚本

功能：
1. 通过股票名称获取股票代码
2. 获取多周期（1m, 5m, 15m, 60m, 日线, 周线）行情数据
3. 每个周期获取 255 个 bar
4. 引用趋势脚本进行趋势分析
5. 引用空间脚本进行止盈止损空间计算
6. 引用背离脚本进行触发信号计算
7. 批量生成股票趋势分析 HTML 文件
8. 定时扫描背离信号并发送飞书通知

Usage:
    # 单只股票生成 HTML
    python quote_snapshot.py --name 兴业银锡
    python quote_snapshot.py --symbol 000426
    
    # 批量生成（遍历 stock.xlsx 中的所有股票）
    python quote_snapshot.py --batch
    python quote_snapshot.py --batch --stock-list /path/to/stock.xlsx --output-dir /path/to/output
    
    # 定时扫描背离信号（每5分钟整点触发，仅在交易时段运行）
    python quote_snapshot.py --schedule
    nohup python src/quote_snapshot.py --schedule > scan.log 2>&1 &
"""
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 配置日志
logger = logging.getLogger('quote_snapshot')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.pytdx_client import (
    connect_pytdx,
    get_kline_data,
    get_multi_period_kline,
    get_stock_code_by_name,
    get_stock_name,
    market_from_code,
    TdxHq_API,
)

from app.trend_script import (
    TrendConfig,
    TrendOutput,
    trend_script,
)
from app.space_script import (
    SpaceConfig,
    SpaceOutput,
    SpaceZone,
    space_script,
    _get_lz_bounds,
    _calc_space_metrics,
    _calc_bollinger_bands,
)
from cores.amp_plotly import (
    AMPConfig,
    compute_amp_last,
    fetch_daily_pytdx,
)
from cores.dynamic_swing_anchored_vwap import (
    DSAConfig,
    dynamic_swing_anchored_vwap,
    fetch_kline_pytdx,
)
from cores.liquidity_zones_plotly import (
    LZConfig,
    build_liquidity_zones,
    atr_wilder,
)
from cores.divergence_many_plotly import (
    DivConfig,
    compute_indicators,
    pivots_confirmed,
    calculate_divs,
    _line_dash,
)


PYTDX_SERVERS = [
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


def compute_trend_for_period(df: pd.DataFrame, period: str, cfg: TrendConfig = None) -> Dict:
    """计算指定周期的趋势分析"""
    if cfg is None:
        cfg = TrendConfig()
    
    df_copy = df.copy()
    df_copy = df_copy.set_index('datetime')
    
    amp_cfg = AMPConfig(
        useAdaptive=cfg.amp_adaptive,
        pI=cfg.amp_period,
        uL=cfg.amp_use_log,
    )
    amp_result = compute_amp_last(df_copy, amp_cfg)
    amp_metrics = amp_result["metrics"]
    
    amp_strength = amp_metrics["strength_pR_cD"]
    amp_position = amp_metrics["close_pos_0_1"]
    
    dsa_cfg = DSAConfig(
        prd=cfg.dsa_prd,
        baseAPT=cfg.dsa_base_apt,
        useAdapt=cfg.dsa_use_adapt,
    )
    vwap_series, dir_series, pivot_labels, segments = dynamic_swing_anchored_vwap(df_copy, dsa_cfg)
    
    last_close = float(df_copy["close"].iloc[-1])
    last_vwap = float(vwap_series.iloc[-1]) if pd.notna(vwap_series.iloc[-1]) else last_close
    last_dir = int(dir_series.iloc[-1]) if pd.notna(dir_series.iloc[-1]) else 0
    
    vwap_position = 0.5
    if last_vwap > 0:
        recent_high = float(df_copy["high"].tail(50).max())
        recent_low = float(df_copy["low"].tail(50).min())
        price_range = recent_high - recent_low
        if price_range > 0:
            vwap_position = (last_vwap - recent_low) / price_range
            vwap_position = float(np.clip(vwap_position, 0.0, 1.0))
    
    def determine_zone_type(position: float, discount_th: float, premium_th: float):
        if position <= discount_th:
            return "discount", 1.0 - position
        elif position >= premium_th:
            return "premium", position
        else:
            return "neutral", 0.5
    
    amp_zone = determine_zone_type(amp_position, cfg.discount_threshold, cfg.premium_threshold)
    vwap_zone = determine_zone_type(vwap_position, cfg.discount_threshold, cfg.premium_threshold)
    
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
    
    strength_threshold_raw = cfg.strength_threshold * 2.0 - 1.0
    if amp_strength >= strength_threshold_raw:
        if last_dir > 0:
            trend_dir = 1
        elif last_dir < 0:
            trend_dir = -1
        else:
            trend_dir = 1 if amp_position < 0.5 else (-1 if amp_position > 0.5 else 0)
    else:
        trend_dir = 0
    
    return {
        'period': period,
        'trend_dir': trend_dir,
        'trend_strength': amp_strength,
        'trend_position': amp_position,
        'zone_type': amp_zone[0],
        'zone_score': amp_zone[1],
        'amp_active_upper': amp_active_upper,
        'amp_active_lower': amp_active_lower,
        'amp_metrics': amp_metrics,
        'amp_result': amp_result,
        'vwap_series': vwap_series,
        'dir_series': dir_series,
        'pivot_labels': pivot_labels,
        'segments': segments,
        'last_vwap': last_vwap,
        'last_close': last_close,
        'vwap_position': vwap_position,
        'df': df_copy.reset_index(),
    }


def compute_space_for_period(df: pd.DataFrame, period: str, cfg: SpaceConfig = None) -> Dict:
    """计算指定周期的空间分析"""
    if cfg is None:
        cfg = SpaceConfig()
    
    df_copy = df.copy()
    df_copy = df_copy.set_index('datetime')
    
    last_close = float(df_copy["close"].iloc[-1])
    atr = atr_wilder(df_copy, 200).iloc[-1]
    if pd.isna(atr):
        atr = atr_wilder(df_copy, 50).iloc[-1]
    if pd.isna(atr):
        atr = 0.0
    
    lz_upper, lz_lower, lz_zones = _get_lz_bounds(df_copy, cfg)
    
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
    
    bb_upper, bb_mid, bb_lower = _calc_bollinger_bands(df_copy, cfg.bb_period, cfg.bb_std)
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
    
    return {
        'period': period,
        'lz_space': lz_space,
        'bb_space': bb_space,
        'lz_zones': lz_zones,
        'last_close': last_close,
        'atr': float(atr) if pd.notna(atr) else 0.0,
        'bb_upper': bb_upper,
        'bb_mid': bb_mid,
        'bb_lower': bb_lower,
    }


def compute_volume_zscore(df: pd.DataFrame, win: int = 255) -> Dict:
    """计算成交量 ZScore
    
    z = (volume - rolling_mean(volume, win)) / rolling_std(volume, win)
    
    返回:
        z: ZScore 序列
        mu: rolling mean
        sd: rolling std
        last_z: 最后一根 bar 的 ZScore
    """
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    
    vol = df["volume"].astype(float)
    mu = vol.rolling(win, min_periods=win).mean()
    sd = vol.rolling(win, min_periods=win).std(ddof=0)
    z = (vol - mu) / sd.replace(0.0, np.nan)
    
    last_z = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else np.nan
    last_mu = float(mu.iloc[-1]) if pd.notna(mu.iloc[-1]) else np.nan
    last_sd = float(sd.iloc[-1]) if pd.notna(sd.iloc[-1]) else np.nan
    last_vol = float(vol.iloc[-1])
    
    return {
        'z': z,
        'mu': mu,
        'sd': sd,
        'last_z': last_z,
        'last_mu': last_mu,
        'last_sd': last_sd,
        'last_vol': last_vol,
        'win': win,
    }


def compute_divergence_for_period(df: pd.DataFrame, period: str, cfg: DivConfig = None) -> Dict:
    """计算指定周期的背离特征
    
    返回:
        pos_lines: 正背离（底背离）线条列表
        neg_lines: 负背离（顶背离）线条列表
        pos_labels: 正背离标签列表
        neg_labels: 负背离标签列表
        last_feat: 最后一根 bar 的背离特征
    """
    if cfg is None:
        cfg = DivConfig(
            prd=5,
            source="Close",
            searchdiv="Regular",
            showlimit=1,
            maxpp=10,
            maxbars=100,
            dontconfirm=False,
            showlines=True,
            showpivot=False,
            calcmacd=True,
            calcmacda=True,
            calcrsi=False,
            calcstoc=False,
            calccci=False,
            calcmom=False,
            calcobv=True,
            calcvwmacd=False,
            calccmf=False,
            calcmfi=False,
            calcext=False,
        )
    
    df_copy = df.copy()
    if 'datetime' in df_copy.columns:
        df_copy = df_copy.set_index('datetime')
    
    n = len(df_copy)
    if n < 60:
        return {
            'period': period,
            'pos_lines': [],
            'neg_lines': [],
            'pos_labels': [],
            'neg_labels': [],
            'last_feat': {},
        }
    
    ind = compute_indicators(df_copy)
    
    close = df_copy["close"].to_numpy(dtype=float)
    high = df_copy["high"].to_numpy(dtype=float)
    low = df_copy["low"].to_numpy(dtype=float)
    
    pivot_src_high = (df_copy["close"] if cfg.source == "Close" else df_copy["high"]).to_numpy(dtype=float)
    pivot_src_low = (df_copy["close"] if cfg.source == "Close" else df_copy["low"]).to_numpy(dtype=float)
    
    ph_conf, _ = pivots_confirmed(pivot_src_high, cfg.prd)
    _, pl_conf = pivots_confirmed(pivot_src_low, cfg.prd)
    
    maxarraysize = 20
    ph_positions: List[int] = []
    pl_positions: List[int] = []
    ph_vals: List[float] = []
    pl_vals: List[float] = []
    
    pos_div_lines: List[Dict] = []
    neg_div_lines: List[Dict] = []
    pos_div_labels: List[Tuple] = []
    neg_div_labels: List[Tuple] = []
    
    last_pos_div_lines = 0
    last_neg_div_lines = 0
    remove_last_pos_divs = False
    remove_last_neg_divs = False
    
    div_events: List[Tuple[int, str]] = []
    last_feat: Dict = {}
    
    # 记录每个时间点的背离指标
    div_indicators_at_time: Dict[int, str] = {}
    
    startpoint = 0 if cfg.dontconfirm else 1
    
    xcat = [str(idx) for idx in df_copy.index]
    
    div_colors = [
        cfg.pos_reg_div_col,
        cfg.neg_reg_div_col,
        cfg.pos_hid_div_col,
        cfg.neg_hid_div_col,
    ]
    
    indicator_names = ["MACD", "Hist", "OBV"]
    indicator_keys = ["macd", "hist", "obv"]
    
    for t in range(n):
        if np.isfinite(ph_conf[t]):
            ph_positions.insert(0, t)
            ph_vals.insert(0, float(ph_conf[t]))
            if len(ph_positions) > maxarraysize:
                ph_positions.pop()
                ph_vals.pop()
        
        if np.isfinite(pl_conf[t]):
            pl_positions.insert(0, t)
            pl_vals.insert(0, float(pl_conf[t]))
            if len(pl_positions) > maxarraysize:
                pl_positions.pop()
                pl_vals.pop()
        
        if np.isfinite(pl_conf[t]):
            remove_last_pos_divs = False
            last_pos_div_lines = 0
        if np.isfinite(ph_conf[t]):
            remove_last_neg_divs = False
            last_neg_div_lines = 0
        
        all_divs = []
        for key in indicator_keys:
            if key not in ind:
                all_divs.extend([0, 0, 0, 0])
                continue
            src_arr = ind[key].to_numpy(dtype=float)
            divs4 = calculate_divs(
                enabled=True,
                src=src_arr,
                close=close, high=high, low=low,
                ph_positions=ph_positions, ph_vals=ph_vals,
                pl_positions=pl_positions, pl_vals=pl_vals,
                t=t, cfg=cfg
            )
            all_divs.extend(divs4)
        
        total_div = sum(1 for v in all_divs if v > 0)
        if total_div < cfg.showlimit:
            all_divs = [0] * len(all_divs)
        
        divergence_text_top = ""
        divergence_text_bottom = ""
        distances: set = set()
        dnumdiv_top = 0
        dnumdiv_bottom = 0
        top_label_col = "rgba(255,255,255,1.0)"
        bottom_label_col = "rgba(255,255,255,1.0)"
        old_pos_divs_can_be_removed = True
        old_neg_divs_can_be_removed = True
        
        for x, key in enumerate(indicator_keys):
            div_type = -1
            for y in range(4):
                dist = all_divs[x * 4 + y]
                if dist > 0:
                    div_type = y
                    if (y % 2) == 1:
                        dnumdiv_top += 1
                        top_label_col = div_colors[y]
                    else:
                        dnumdiv_bottom += 1
                        bottom_label_col = div_colors[y]
                    
                    if dist not in distances:
                        distances.add(dist)
                        
                        if cfg.showlines:
                            x1 = xcat[t - dist] if (t - dist) >= 0 else xcat[0]
                            x2 = xcat[t - startpoint] if (t - startpoint) >= 0 else xcat[t]
                            
                            if cfg.source == "Close":
                                y1 = float(close[t - dist]) if (t - dist) >= 0 else float(close[0])
                                y2 = float(close[t - startpoint]) if (t - startpoint) >= 0 else float(close[t])
                            else:
                                if (y % 2) == 0:
                                    y1 = float(low[t - dist]) if (t - dist) >= 0 else float(low[0])
                                    y2 = float(low[t - startpoint]) if (t - startpoint) >= 0 else float(low[t])
                                else:
                                    y1 = float(high[t - dist]) if (t - dist) >= 0 else float(high[0])
                                    y2 = float(high[t - startpoint]) if (t - startpoint) >= 0 else float(high[t])
                            
                            dash = _line_dash(cfg.reg_div_l_style if y < 2 else cfg.hid_div_l_style)
                            width = cfg.reg_div_l_width if y < 2 else cfg.hid_div_l_width
                            new_line = dict(
                                x=[x1, x2], y=[y1, y2],
                                color=div_colors[y], width=width, dash=dash,
                                is_pos=((y % 2) == 0)
                            )
                            
                            if (y % 2) == 0:
                                if old_pos_divs_can_be_removed:
                                    old_pos_divs_can_be_removed = False
                                    if (not cfg.showlast) and remove_last_pos_divs:
                                        for _ in range(last_pos_div_lines):
                                            if pos_div_lines:
                                                pos_div_lines.pop()
                                        last_pos_div_lines = 0
                                    if cfg.showlast:
                                        pos_div_lines.clear()
                                pos_div_lines.append(new_line)
                                last_pos_div_lines += 1
                                remove_last_pos_divs = True
                            else:
                                if old_neg_divs_can_be_removed:
                                    old_neg_divs_can_be_removed = False
                                    if (not cfg.showlast) and remove_last_neg_divs:
                                        for _ in range(last_neg_div_lines):
                                            if neg_div_lines:
                                                neg_div_lines.pop()
                                        last_neg_div_lines = 0
                                    if cfg.showlast:
                                        neg_div_lines.clear()
                                neg_div_lines.append(new_line)
                                last_neg_div_lines += 1
                                remove_last_neg_divs = True
            
            if div_type >= 0 and x < len(indicator_names):
                if (div_type % 2) == 1:
                    divergence_text_top += indicator_names[x] + "\n"
                else:
                    divergence_text_bottom += indicator_names[x] + "\n"
                
                # 记录该时间点的背离指标
                if t not in div_indicators_at_time:
                    div_indicators_at_time[t] = indicator_names[x]
        
        if dnumdiv_top > 0:
            div_events.append((t, "top"))
        if dnumdiv_bottom > 0:
            div_events.append((t, "bottom"))
        
        if cfg.shownum and dnumdiv_top > 0:
            divergence_text_top += str(dnumdiv_top)
        if cfg.shownum and dnumdiv_bottom > 0:
            divergence_text_bottom += str(dnumdiv_bottom)
        
        if divergence_text_top.strip():
            if cfg.showlast:
                neg_div_labels.clear()
            y_top = float(max(high[t], high[t-1])) if t-1 >= 0 else float(high[t])
            neg_div_labels.append((xcat[t], y_top, divergence_text_top.strip(), top_label_col, cfg.neg_div_text_col))
        
        if divergence_text_bottom.strip():
            if cfg.showlast:
                pos_div_labels.clear()
            y_bot = float(min(low[t], low[t-1])) if t-1 >= 0 else float(low[t])
            pos_div_labels.append((xcat[t], y_bot, divergence_text_bottom.strip(), bottom_label_col, cfg.pos_div_text_col))
    
    if div_events:
        last_t = n - 1
        last_n = div_events[-5:] if len(div_events) >= 5 else div_events
        top_cnt = sum(1 for _, d in last_n if d == "top")
        bottom_cnt = sum(1 for _, d in last_n if d == "bottom")
        div_bias = (bottom_cnt - top_cnt) / max(len(last_n), 1)
        
        last_event_t = div_events[-1][0]
        dirs_at_last = {d for (t, d) in div_events if t == last_event_t}
        if dirs_at_last == {"top"}:
            last_type = "top"
        elif dirs_at_last == {"bottom"}:
            last_type = "bottom"
        else:
            last_type = "conflict"
        
        last_age = int(last_t - last_event_t)
        
        M = 100
        win_start = max(last_t - M + 1, 0)
        ev_in_window = sum(1 for (t, _) in div_events if win_start <= t <= last_t)
        div_rate = ev_in_window / M
        
        # 获取背离发生的时间点
        div_last_time = str(xcat[last_event_t]) if last_event_t < len(xcat) else ""
        
        # 获取最后一个背离的指标类型（从记录的字典中获取）
        div_last_indicator = div_indicators_at_time.get(last_event_t, "")
        
        last_feat = {
            "div_bias": float(div_bias),
            "div_last_type": last_type,
            "div_last_age": last_age,
            "div_last_age_norm": min(last_age, 100) / 100,
            "div_rate": float(div_rate),
            "div_last_time": div_last_time,
            "div_last_indicator": div_last_indicator,  # 添加指标类型
        }
    
    return {
        'period': period,
        'pos_lines': pos_div_lines,
        'neg_lines': neg_div_lines,
        'pos_labels': pos_div_labels,
        'neg_labels': neg_div_labels,
        'last_feat': last_feat,
        'df': df_copy,
    }


def generate_divergence_chart_image(symbol: str, stock_name: str, div_data: Dict, output_dir: str, space_data: Dict = None) -> str:
    """生成背离图表图片（60m, 15m, 5m, 1m 合成一张图）
    
    Args:
        symbol: 股票代码
        stock_name: 股票名称
        div_data: 背离数据字典
        output_dir: 输出目录
        space_data: 空间数据字典（包含 LZ 和 BB 上下界）
    
    Returns:
        图片文件路径
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from PIL import Image
    import io
    
    periods = ['60m', '15m', '5m', '1m']
    
    TV_BG = '#131722'
    TV_GRID = 'rgba(42, 46, 57, 0.5)'
    TV_TEXT = '#d1d4dc'
    TV_UP = '#ef5350'
    TV_DOWN = '#26a69a'
    
    display_name = f"{stock_name} ({symbol})" if stock_name else symbol
    
    # 为每个周期生成图片
    img_paths = []
    
    for period in periods:
        if not div_data or period not in div_data:
            continue
        
        div = div_data[period]
        div_df = div.get('df')
        if div_df is None or div_df.empty:
            continue
        
        if 'datetime' in div_df.columns:
            div_df_indexed = div_df.set_index('datetime')
        else:
            div_df_indexed = div_df
        
        x_str = [str(idx) for idx in div_df_indexed.index]
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
        )
        
        fig.add_trace(
            go.Candlestick(
                x=x_str,
                open=div_df_indexed["open"],
                high=div_df_indexed["high"],
                low=div_df_indexed["low"],
                close=div_df_indexed["close"],
                increasing_line_color=TV_UP,
                decreasing_line_color=TV_DOWN,
                increasing_fillcolor=TV_UP,
                decreasing_fillcolor=TV_DOWN,
                showlegend=False,
            ),
            row=1, col=1,
        )
        
        # 60m 周期添加布林带和上下界
        if period == '60m' and space_data and '60m' in space_data:
            sp_60m = space_data['60m']
            bb_upper = sp_60m.get('bb_upper')
            bb_mid = sp_60m.get('bb_mid')
            bb_lower = sp_60m.get('bb_lower')
            lz_space = sp_60m.get('lz_space')
            
            # 布林带
            if bb_upper is not None and len(bb_upper) == len(x_str):
                fig.add_trace(
                    go.Scatter(x=x_str, y=bb_upper, mode="lines",
                               line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash"),
                               name="BB Upper", showlegend=False),
                    row=1, col=1,
                )
            if bb_lower is not None and len(bb_lower) == len(x_str):
                fig.add_trace(
                    go.Scatter(x=x_str, y=bb_lower, mode="lines",
                               line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash"),
                               name="BB Lower", showlegend=False),
                    row=1, col=1,
                )
            if bb_mid is not None and len(bb_mid) == len(x_str):
                fig.add_trace(
                    go.Scatter(x=x_str, y=bb_mid, mode="lines",
                               line=dict(color="rgba(255,193,7,0.4)", width=1),
                               name="BB Mid", showlegend=False),
                    row=1, col=1,
                )
            
            # 日线上下界
            if space_data.get('d'):
                d_space = space_data['d'].get('lz_space')
                d_bb_space = space_data['d'].get('bb_space')
                if d_space:
                    if d_space.upper_bound:
                        fig.add_hline(
                            y=d_space.upper_bound,
                            line=dict(color="#ff9800", width=2, dash="dot"),
                            annotation=dict(text=f"日线LZ上界 {d_space.upper_bound:.2f}", font=dict(size=10, color="#ff9800")),
                            row=1,
                        )
                    if d_space.lower_bound:
                        fig.add_hline(
                            y=d_space.lower_bound,
                            line=dict(color="#2196f3", width=2, dash="dot"),
                            annotation=dict(text=f"日线LZ下界 {d_space.lower_bound:.2f}", font=dict(size=10, color="#2196f3")),
                            row=1,
                        )
                if d_bb_space:
                    if d_bb_space.upper_bound:
                        fig.add_hline(
                            y=d_bb_space.upper_bound,
                            line=dict(color="#e91e63", width=1, dash="dash"),
                            annotation=dict(text=f"日线BB上界 {d_bb_space.upper_bound:.2f}", font=dict(size=9, color="#e91e63")),
                            row=1,
                        )
                    if d_bb_space.lower_bound:
                        fig.add_hline(
                            y=d_bb_space.lower_bound,
                            line=dict(color="#9c27b0", width=1, dash="dash"),
                            annotation=dict(text=f"日线BB下界 {d_bb_space.lower_bound:.2f}", font=dict(size=9, color="#9c27b0")),
                            row=1,
                        )
            
            # 60m LZ 上界
            if lz_space and lz_space.upper_bound:
                fig.add_hline(
                    y=lz_space.upper_bound,
                    line=dict(color="#2370a3", width=2, dash="dash"),
                    annotation=dict(text=f"60m LZ上界 {lz_space.upper_bound:.2f}", font=dict(size=10, color="#2370a3")),
                    row=1,
                )
        
        # 背离线条和标签
        pos_lines = div['pos_lines']
        neg_lines = div['neg_lines']
        pos_labels = div['pos_labels']
        neg_labels = div['neg_labels']
        
        for ln in pos_lines:
            if isinstance(ln, dict):
                fig.add_trace(
                    go.Scatter(
                        x=ln["x"], y=ln["y"],
                        mode="lines",
                        line=dict(width=ln["width"], dash=ln["dash"]),
                        marker=dict(color=ln["color"]),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=ln[0], y=ln[1],
                        mode="lines",
                        line=dict(width=ln[3] if len(ln) > 3 else 1, dash=ln[4] if len(ln) > 4 else "solid"),
                        marker=dict(color=ln[2] if len(ln) > 2 else "#26a69a"),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        for ln in neg_lines:
            if isinstance(ln, dict):
                fig.add_trace(
                    go.Scatter(
                        x=ln["x"], y=ln["y"],
                        mode="lines",
                        line=dict(width=ln["width"], dash=ln["dash"]),
                        marker=dict(color=ln["color"]),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=ln[0], y=ln[1],
                        mode="lines",
                        line=dict(width=ln[3] if len(ln) > 3 else 1, dash=ln[4] if len(ln) > 4 else "solid"),
                        marker=dict(color=ln[2] if len(ln) > 2 else "#ef5350"),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        for lab in pos_labels:
            if isinstance(lab, dict):
                fig.add_annotation(
                    x=lab["x"], y=lab["y"], xref="x", yref="y", text=lab["text"],
                    showarrow=True, arrowhead=2, ax=0, ay=-25,
                    bgcolor=lab.get("bgcolor", "rgba(0,230,118,0.85)"), font=dict(color="white", size=10),
                    bordercolor=lab.get("bgcolor", "rgba(0,230,118,0.85)"), borderwidth=1, row=1, col=1,
                )
            else:
                fig.add_annotation(
                    x=lab[0], y=lab[1], xref="x", yref="y", text=lab[2],
                    showarrow=True, arrowhead=2, ax=0, ay=-25,
                    bgcolor=lab[3] if len(lab) > 3 else "rgba(0,230,118,0.85)", font=dict(color="white", size=10),
                    bordercolor=lab[3] if len(lab) > 3 else "rgba(0,230,118,0.85)", borderwidth=1, row=1, col=1,
                )
        
        for lab in neg_labels:
            if isinstance(lab, dict):
                fig.add_annotation(
                    x=lab["x"], y=lab["y"], xref="x", yref="y", text=lab["text"],
                    showarrow=True, arrowhead=2, ax=0, ay=25,
                    bgcolor=lab.get("bgcolor", "rgba(255,23,68,0.85)"), font=dict(color="white", size=10),
                    bordercolor=lab.get("bgcolor", "rgba(255,23,68,0.85)"), borderwidth=1, row=1, col=1,
                )
            else:
                fig.add_annotation(
                    x=lab[0], y=lab[1], xref="x", yref="y", text=lab[2],
                    showarrow=True, arrowhead=2, ax=0, ay=25,
                    bgcolor=lab[3] if len(lab) > 3 else "rgba(255,23,68,0.85)", font=dict(color="white", size=10),
                    bordercolor=lab[3] if len(lab) > 3 else "rgba(255,23,68,0.85)", borderwidth=1, row=1, col=1,
                )
        
        # 成交量
        vol_colors = np.where(div_df_indexed["close"] >= div_df_indexed["open"],
                              "rgba(239,83,80,0.6)", "rgba(38,166,154,0.6)")
        fig.add_trace(
            go.Bar(x=x_str, y=div_df_indexed["volume"], marker_color=vol_colors, showlegend=False),
            row=2, col=1,
        )
        
        # 背离特征信息
        last_feat = div.get('last_feat')
        div_info = ""
        if last_feat:
            div_type = last_feat.get('div_last_type', 'none')
            div_age = last_feat.get('div_last_age', 999)
            div_bias = last_feat.get('div_bias', 0)
            if div_type != 'none':
                type_text = "顶背离" if div_type == 'top' else "底背离"
                div_info = f"<b>{period} {type_text}</b><br>age: {div_age} bars<br>bias: {div_bias:.3f}"
        
        if div_info:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
                text=div_info, align="left", showarrow=False,
                font=dict(size=11, color="#c9d1d9"),
                bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1, borderpad=8,
            )
        
        fig.update_layout(
            title=dict(text=f"{display_name} - {period}", font=dict(size=16, color=TV_TEXT)),
            plot_bgcolor=TV_BG, paper_bgcolor=TV_BG,
            font=dict(color=TV_TEXT), height=700, width=1400, margin=dict(l=50, r=50, t=60, b=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor=TV_GRID, rangeslider_visible=False, type="category", categoryorder="array", categoryarray=x_str)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=2, col=1, rangemode="tozero")
        
        # 保存为图片（高分辨率）
        img_path = os.path.join(output_dir, f"{stock_name}_{period}_div.png")
        fig.write_image(img_path, scale=4)
        img_paths.append(img_path)
    
    if not img_paths:
        return None
    
    # 合成一张图片
    if len(img_paths) == 1:
        return img_paths[0]
    
    # 垂直拼接所有图片
    images = [Image.open(p) for p in img_paths]
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    
    combined = Image.new('RGB', (max_width, total_height), color=(19, 23, 34))
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    
    combined_path = os.path.join(output_dir, f"{stock_name}_divergence_combined.png")
    combined.save(combined_path, quality=95, optimize=False)
    
    # 删除临时图片
    for p in img_paths:
        if os.path.exists(p):
            os.remove(p)
    
    return combined_path


def build_trend_tab_html(data: Dict, symbol: str, stock_name: str, output_path: str, space_data: Dict = None, div_data: Dict = None, vol_zscore_data: Dict = None):
    """生成带Tab的HTML报告（周线、日线趋势分析 + 60m空间分析）"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    display_name = f"{stock_name} ({symbol})" if stock_name else symbol
    
    TV_BG = '#131722'
    TV_GRID = 'rgba(42, 46, 57, 0.5)'
    TV_TEXT = '#d1d4dc'
    TV_UP = '#ef5350'
    TV_DOWN = '#26a69a'
    
    plot_divs = []
    
    # Tab 1: 周线趋势
    if 'w' in data:
        trend_data = data['w']
        df = trend_data['df']
        amp_result = trend_data['amp_result']
        amp_metrics = trend_data['amp_metrics']
        pivot_labels = trend_data['pivot_labels']
        segments = trend_data['segments']
        
        ch = amp_result["channel"]
        cfg = amp_result["cfg"]
        
        df_indexed = df.set_index('datetime')
        x_str = [str(idx) for idx in df_indexed.index]
        
        has_vol_z_w = vol_zscore_data and 'w' in vol_zscore_data
        if has_vol_z_w:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.60, 0.20, 0.20],
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.75, 0.25],
            )
        
        fig.add_trace(
            go.Candlestick(
                x=x_str,
                open=df_indexed["open"],
                high=df_indexed["high"],
                low=df_indexed["low"],
                close=df_indexed["close"],
                increasing_line_color=TV_UP,
                decreasing_line_color=TV_DOWN,
                increasing_fillcolor=TV_UP,
                decreasing_fillcolor=TV_DOWN,
                showlegend=False,
            ),
            row=1, col=1,
        )
        
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["upper"], mode="lines",
                       line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["lower"], mode="lines",
                       line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["upper"], mode="lines",
                       line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["lower"], mode="lines", fill="tonexty",
                       fillcolor=cfg.channelFill, line=dict(color="rgba(0,0,0,0)"),
                       hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        
        for al in amp_result.get("activityLines", []):
            fig.add_trace(
                go.Scatter(x=[str(x) for x in al["x"]], y=al["y"], mode="lines",
                           line=dict(color=al["color"], width=1), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )
        
        for seg in segments:
            seg_dir = seg["dir"]
            color = "#ff1744" if seg_dir > 0 else "#00e676"
            fig.add_trace(
                go.Scatter(x=[str(x) for x in seg["x"]], y=seg["y"], mode="lines",
                           line=dict(width=2, color=color), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )
        
        for lab in pivot_labels:
            lab_x_str = str(lab["x"])
            if lab_x_str not in x_str:
                continue
            txt = lab["text"]
            if not txt:
                continue
            is_up = (lab["dir"] > 0)
            bgcolor = "rgba(0,230,118,0.85)" if is_up else "rgba(255,23,68,0.85)"
            ay = 25 if is_up else -25
            fig.add_annotation(
                x=lab_x_str, y=lab["y"], xref="x", yref="y", text=txt,
                showarrow=True, arrowhead=2, ax=0, ay=ay,
                bgcolor=bgcolor, font=dict(color="white", size=12),
                bordercolor=bgcolor, borderwidth=1, row=1, col=1,
            )
        
        vol_colors = np.where(df_indexed["close"] >= df_indexed["open"],
                              "rgba(239,83,80,0.6)", "rgba(38,166,154,0.6)")
        fig.add_trace(
            go.Bar(x=x_str, y=df_indexed["volume"], marker_color=vol_colors, showlegend=False),
            row=2, col=1,
        )
        
        # Volume ZScore (柱状图)
        vol_z_info_w = ""
        if has_vol_z_w:
            vz = vol_zscore_data['w']
            z_series = vz['z']
            z_plot = z_series.clip(lower=-6.0, upper=6.0)
            # 重置索引确保与 x_str 对齐
            z_vals = [float(v) if pd.notna(v) else 0 for v in z_plot.values]
            # 根据正负值设置颜色
            z_colors = ["rgba(38,166,154,0.7)" if v >= 0 else "rgba(239,83,80,0.7)" for v in z_vals]
            fig.add_trace(
                go.Bar(x=x_str, y=z_vals, marker_color=z_colors, showlegend=False),
                row=3, col=1,
            )
            for y0 in [0, 1, -1, 2, -2]:
                fig.add_hline(y=y0, row=3, col=1, line_width=1, opacity=0.35)
            
            last_z = vz['last_z']
            vol_z_info_w = f"<br><b>成交量ZScore</b><br>z: {last_z:.3f}<br>" if np.isfinite(last_z) else ""
        
        dir_text = {1: "↑ 上涨", -1: "↓ 下跌", 0: "→ 震荡"}[trend_data['trend_dir']]
        zone_text = {"discount": "折价区", "neutral": "中性区", "premium": "溢价区"}[trend_data['zone_type']]
        amp_upper_str = f"{trend_data['amp_active_upper']:.2f}" if trend_data['amp_active_upper'] else "N/A"
        amp_lower_str = f"{trend_data['amp_active_lower']:.2f}" if trend_data['amp_active_lower'] else "N/A"
        
        info_html = (
            f"<b>W 周期趋势分析</b><br>"
            f"趋势方向: {dir_text}<br>"
            f"趋势强度: {trend_data['trend_strength']:.4f}<br>"
            f"当前位置: {trend_data['trend_position']:.4f}<br>"
            f"<br><b>最佳交易区</b><br>"
            f"类型: {zone_text}<br>"
            f"得分: {trend_data['zone_score']:.4f}<br>"
            f"活跃区上界: {amp_upper_str}<br>"
            f"活跃区下界: {amp_lower_str}<br>"
            f"{vol_z_info_w}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
            text=info_html, align="left", showarrow=False,
            font=dict(size=12, color="#c9d1d9"),
            bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1, borderpad=8,
        )
        
        shapes = []
        for poly in amp_result.get("profilePolys", []):
            x0, x1 = str(poly["x0"]), str(poly["x1"])
            y0t, y0b = poly["y0_top"], poly["y0_bot"]
            y1t, y1b = poly["y1_top"], poly["y1_bot"]
            shapes.append(dict(
                type="path", xref="x", yref="y",
                path=f"M {x0} {y0t} L {x0} {y0b} L {x1} {y1b} L {x1} {y1t} Z",
                fillcolor=poly["fill"], line=dict(width=0), layer="below", opacity=1.0,
            ))
        
        fig_height_w = 980 if has_vol_z_w else 800
        fig.update_layout(
            shapes=shapes, plot_bgcolor=TV_BG, paper_bgcolor=TV_BG,
            font=dict(color=TV_TEXT), height=fig_height_w, margin=dict(l=40, r=40, t=40, b=40),
        )
        fig.update_xaxes(showgrid=True, gridcolor=TV_GRID, rangeslider_visible=False, type="category", categoryorder="array", categoryarray=x_str)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=2, col=1, rangemode="tozero")
        if has_vol_z_w:
            fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=3, col=1)
        
        plot_divs.append(('w', fig.to_html(full_html=False, include_plotlyjs=False)))
    
    # Tab 2: 日线趋势
    if 'd' in data:
        trend_data = data['d']
        df = trend_data['df']
        amp_result = trend_data['amp_result']
        amp_metrics = trend_data['amp_metrics']
        pivot_labels = trend_data['pivot_labels']
        segments = trend_data['segments']
        
        ch = amp_result["channel"]
        cfg = amp_result["cfg"]
        
        df_indexed = df.set_index('datetime')
        x_str = [str(idx) for idx in df_indexed.index]
        
        has_vol_z = vol_zscore_data and 'd' in vol_zscore_data
        if has_vol_z:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.60, 0.20, 0.20],
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.75, 0.25],
            )
        
        fig.add_trace(
            go.Candlestick(
                x=x_str,
                open=df_indexed["open"],
                high=df_indexed["high"],
                low=df_indexed["low"],
                close=df_indexed["close"],
                increasing_line_color=TV_UP,
                decreasing_line_color=TV_DOWN,
                increasing_fillcolor=TV_UP,
                decreasing_fillcolor=TV_DOWN,
                showlegend=False,
            ),
            row=1, col=1,
        )
        
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["upper"], mode="lines",
                       line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["lower"], mode="lines",
                       line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["upper"], mode="lines",
                       line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[str(x) for x in ch["x"]], y=ch["lower"], mode="lines", fill="tonexty",
                       fillcolor=cfg.channelFill, line=dict(color="rgba(0,0,0,0)"),
                       hoverinfo="skip", showlegend=False),
            row=1, col=1,
        )
        
        for al in amp_result.get("activityLines", []):
            fig.add_trace(
                go.Scatter(x=[str(x) for x in al["x"]], y=al["y"], mode="lines",
                           line=dict(color=al["color"], width=1), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )
        
        for seg in segments:
            seg_dir = seg["dir"]
            color = "#ff1744" if seg_dir > 0 else "#00e676"
            fig.add_trace(
                go.Scatter(x=[str(x) for x in seg["x"]], y=seg["y"], mode="lines",
                           line=dict(width=2, color=color), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )
        
        for lab in pivot_labels:
            lab_x_str = str(lab["x"])
            if lab_x_str not in x_str:
                continue
            txt = lab["text"]
            if not txt:
                continue
            is_up = (lab["dir"] > 0)
            bgcolor = "rgba(0,230,118,0.85)" if is_up else "rgba(255,23,68,0.85)"
            ay = 25 if is_up else -25
            fig.add_annotation(
                x=lab_x_str, y=lab["y"], xref="x", yref="y", text=txt,
                showarrow=True, arrowhead=2, ax=0, ay=ay,
                bgcolor=bgcolor, font=dict(color="white", size=12),
                bordercolor=bgcolor, borderwidth=1, row=1, col=1,
            )
        
        vol_colors = np.where(df_indexed["close"] >= df_indexed["open"],
                              "rgba(239,83,80,0.6)", "rgba(38,166,154,0.6)")
        fig.add_trace(
            go.Bar(x=x_str, y=df_indexed["volume"], marker_color=vol_colors, showlegend=False),
            row=2, col=1,
        )
        
        # Volume ZScore (柱状图)
        vol_z_info = ""
        if has_vol_z:
            vz = vol_zscore_data['d']
            z_series = vz['z']
            z_plot = z_series.clip(lower=-6.0, upper=6.0)
            # 重置索引确保与 x_str 对齐
            z_vals = [float(v) if pd.notna(v) else 0 for v in z_plot.values]
            # 根据正负值设置颜色
            z_colors = ["rgba(38,166,154,0.7)" if v >= 0 else "rgba(239,83,80,0.7)" for v in z_vals]
            fig.add_trace(
                go.Bar(x=x_str, y=z_vals, marker_color=z_colors, showlegend=False),
                row=3, col=1,
            )
            for y0 in [0, 1, -1, 2, -2]:
                fig.add_hline(y=y0, row=3, col=1, line_width=1, opacity=0.35)
            
            last_z = vz['last_z']
            vol_z_info = f"<br><b>成交量ZScore</b><br>z: {last_z:.3f}<br>" if np.isfinite(last_z) else ""
        
        dir_text = {1: "↑ 上涨", -1: "↓ 下跌", 0: "→ 震荡"}[trend_data['trend_dir']]
        zone_text = {"discount": "折价区", "neutral": "中性区", "premium": "溢价区"}[trend_data['zone_type']]
        amp_upper_str = f"{trend_data['amp_active_upper']:.2f}" if trend_data['amp_active_upper'] else "N/A"
        amp_lower_str = f"{trend_data['amp_active_lower']:.2f}" if trend_data['amp_active_lower'] else "N/A"
        
        info_html = (
            f"<b>D 日线趋势分析</b><br>"
            f"趋势方向: {dir_text}<br>"
            f"趋势强度: {trend_data['trend_strength']:.4f}<br>"
            f"当前位置: {trend_data['trend_position']:.4f}<br>"
            f"<br><b>最佳交易区</b><br>"
            f"类型: {zone_text}<br>"
            f"得分: {trend_data['zone_score']:.4f}<br>"
            f"活跃区上界: {amp_upper_str}<br>"
            f"活跃区下界: {amp_lower_str}<br>"
            f"{vol_z_info}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
            text=info_html, align="left", showarrow=False,
            font=dict(size=12, color="#c9d1d9"),
            bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1, borderpad=8,
        )
        
        shapes = []
        for poly in amp_result.get("profilePolys", []):
            x0, x1 = str(poly["x0"]), str(poly["x1"])
            y0t, y0b = poly["y0_top"], poly["y0_bot"]
            y1t, y1b = poly["y1_top"], poly["y1_bot"]
            shapes.append(dict(
                type="path", xref="x", yref="y",
                path=f"M {x0} {y0t} L {x0} {y0b} L {x1} {y1b} L {x1} {y1t} Z",
                fillcolor=poly["fill"], line=dict(width=0), layer="below", opacity=1.0,
            ))
        
        fig_height = 980 if has_vol_z else 800
        fig.update_layout(
            shapes=shapes, plot_bgcolor=TV_BG, paper_bgcolor=TV_BG,
            font=dict(color=TV_TEXT), height=fig_height, margin=dict(l=40, r=40, t=40, b=40),
        )
        fig.update_xaxes(showgrid=True, gridcolor=TV_GRID, rangeslider_visible=False, type="category", categoryorder="array", categoryarray=x_str)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=2, col=1, rangemode="tozero")
        if has_vol_z:
            fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=3, col=1)
        
        plot_divs.append(('d', fig.to_html(full_html=False, include_plotlyjs=False)))
    
    # Tab 3: 60m 空间分析（包含日线上下界）
    if space_data and '60m' in space_data:
        sp = space_data['60m']
        df = sp['df']
        df_indexed = df.set_index('datetime')
        x_str = [str(idx) for idx in df_indexed.index]
        lz_space = sp['lz_space']
        bb_space = sp['bb_space']
        lz_zones = sp['lz_zones']
        bb_upper = sp['bb_upper']
        bb_mid = sp['bb_mid']
        bb_lower = sp['bb_lower']
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
        )
        
        fig.add_trace(
            go.Candlestick(
                x=x_str,
                open=df_indexed["open"],
                high=df_indexed["high"],
                low=df_indexed["low"],
                close=df_indexed["close"],
                increasing_line_color=TV_UP,
                decreasing_line_color=TV_DOWN,
                increasing_fillcolor=TV_UP,
                decreasing_fillcolor=TV_DOWN,
                showlegend=False,
            ),
            row=1, col=1,
        )
        
        # 布林带
        fig.add_trace(
            go.Scatter(x=x_str, y=bb_upper, mode="lines",
                       line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash"),
                       name="BB Upper", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_str, y=bb_lower, mode="lines",
                       line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash"),
                       name="BB Lower", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_str, y=bb_mid, mode="lines",
                       line=dict(color="rgba(255,193,7,0.4)", width=1),
                       name="BB Mid", showlegend=False),
            row=1, col=1,
        )
        
        # 日线上下界（用不同颜色标记）
        if space_data.get('d'):
            d_space = space_data['d']['lz_space']
            d_bb_space = space_data['d']['bb_space']
            # 日线 LZ 上下界
            if d_space.upper_bound:
                fig.add_hline(
                    y=d_space.upper_bound,
                    line=dict(color="#ff9800", width=2, dash="dot"),
                    annotation=dict(text=f"日线LZ上界 {d_space.upper_bound:.2f}", font=dict(size=10, color="#ff9800")),
                    row=1,
                )
            if d_space.lower_bound:
                fig.add_hline(
                    y=d_space.lower_bound,
                    line=dict(color="#2196f3", width=2, dash="dot"),
                    annotation=dict(text=f"日线LZ下界 {d_space.lower_bound:.2f}", font=dict(size=10, color="#2196f3")),
                    row=1,
                )
            # 日线 BB 上下界
            if d_bb_space.upper_bound:
                fig.add_hline(
                    y=d_bb_space.upper_bound,
                    line=dict(color="#e91e63", width=1, dash="dash"),
                    annotation=dict(text=f"日线BB上界 {d_bb_space.upper_bound:.2f}", font=dict(size=9, color="#e91e63")),
                    row=1,
                )
            if d_bb_space.lower_bound:
                fig.add_hline(
                    y=d_bb_space.lower_bound,
                    line=dict(color="#9c27b0", width=1, dash="dash"),
                    annotation=dict(text=f"日线BB下界 {d_bb_space.lower_bound:.2f}", font=dict(size=9, color="#9c27b0")),
                    row=1,
                )
        
        # 60m LZ上下界
        if lz_space.upper_bound:
            fig.add_hline(
                y=lz_space.upper_bound,
                line=dict(color="#2370a3", width=2, dash="dash"),
                annotation=dict(text=f"60m LZ上界 {lz_space.upper_bound:.2f}", font=dict(size=10)),
                row=1,
            )
        if lz_space.lower_bound:
            fig.add_hline(
                y=lz_space.lower_bound,
                line=dict(color="#23a372", width=2, dash="dash"),
                annotation=dict(text=f"60m LZ下界 {lz_space.lower_bound:.2f}", font=dict(size=10)),
                row=1,
            )
        if lz_space.stop_price:
            fig.add_hline(
                y=lz_space.stop_price,
                line=dict(color="#ef5350", width=1, dash="dash"),
                annotation=dict(text=f"止损 {lz_space.stop_price:.2f}", font=dict(size=9)),
                row=1,
            )
        
        # 背离线条和标签
        if div_data and '60m' in div_data:
            div = div_data['60m']
            pos_lines = div['pos_lines']
            neg_lines = div['neg_lines']
            pos_labels = div['pos_labels']
            neg_labels = div['neg_labels']
            
            for ln in pos_lines:
                fig.add_trace(
                    go.Scatter(
                        x=ln["x"], y=ln["y"],
                        mode="lines",
                        line=dict(width=ln["width"], dash=ln["dash"]),
                        marker=dict(color=ln["color"]),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            for ln in neg_lines:
                fig.add_trace(
                    go.Scatter(
                        x=ln["x"], y=ln["y"],
                        mode="lines",
                        line=dict(width=ln["width"], dash=ln["dash"]),
                        marker=dict(color=ln["color"]),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            for (x, y, text_, bg, tc) in neg_labels:
                fig.add_annotation(
                    x=x, y=y, xref="x", yref="y",
                    text=text_.replace("\n", "<br>"),
                    showarrow=True, arrowhead=2, ax=0, ay=-25,
                    bgcolor=bg, font=dict(color=tc, size=12),
                    bordercolor=bg, borderwidth=1,
                    row=1, col=1
                )
            
            for (x, y, text_, bg, tc) in pos_labels:
                fig.add_annotation(
                    x=x, y=y, xref="x", yref="y",
                    text=text_.replace("\n", "<br>"),
                    showarrow=True, arrowhead=2, ax=0, ay=25,
                    bgcolor=bg, font=dict(color=tc, size=12),
                    bordercolor=bg, borderwidth=1,
                    row=1, col=1
                )
        
        vol_colors = np.where(df_indexed["close"] >= df_indexed["open"],
                              "rgba(239,83,80,0.6)", "rgba(38,166,154,0.6)")
        fig.add_trace(
            go.Bar(x=x_str, y=df_indexed["volume"], marker_color=vol_colors, showlegend=False),
            row=2, col=1,
        )
        
        lz_stop_str = f"{lz_space.stop_price:.2f}" if lz_space.stop_price else "N/A"
        bb_stop_str = f"{bb_space.stop_price:.2f}" if bb_space.stop_price else "N/A"
        
        last_close = sp['last_close']
        atr_60m = sp['atr']
        
        # 计算 4 套盈亏比
        def calc_rr(upper, lower, atr, close):
            if upper and lower and atr > 0:
                stop = lower - 2 * atr
                reward = upper - close
                risk = close - stop
                rr = reward / risk if risk > 0 else 0
                rr_pass = rr >= 2.0 and reward > 0 and risk > 0
                return stop, rr, rr_pass
            return None, 0, False
        
        # 1. 60m BB 盈亏比
        m_bb_stop, m_bb_rr, m_bb_pass = calc_rr(bb_space.upper_bound, bb_space.lower_bound, atr_60m, last_close)
        
        # 2. 60m LZ 盈亏比
        m_lz_stop, m_lz_rr, m_lz_pass = calc_rr(lz_space.upper_bound, lz_space.lower_bound, atr_60m, last_close)
        
        # 日线盈亏比（需要日线 ATR）
        d_lz_rr_info = ""
        d_bb_rr_info = ""
        
        if space_data.get('d'):
            d_lz = space_data['d']['lz_space']
            d_bb = space_data['d']['bb_space']
            d_atr = space_data['d']['atr']
            
            # 3. 日线 BB 盈亏比
            d_bb_stop, d_bb_rr, d_bb_pass = calc_rr(d_bb.upper_bound, d_bb.lower_bound, d_atr, last_close)
            
            # 4. 日线 LZ 盈亏比
            d_lz_stop, d_lz_rr, d_lz_pass = calc_rr(d_lz.upper_bound, d_lz.lower_bound, d_atr, last_close)
            
            d_lz_upper_str = f"{d_lz.upper_bound:.2f}" if d_lz.upper_bound is not None else "N/A"
            d_lz_stop_str = f"{d_lz_stop:.2f}" if d_lz_stop is not None else "N/A"
            d_lz_rr_info = (
                f"<br><b>日线LZ盈亏比</b><br>"
                f"止盈: {d_lz_upper_str}<br>"
                f"止损: {d_lz_stop_str}<br>"
                f"盈亏比: {d_lz_rr:.2f} ({'✓' if d_lz_pass else '✗'})<br>"
            ) if d_lz_stop else ""
            d_bb_upper_str = f"{d_bb.upper_bound:.2f}" if d_bb.upper_bound is not None else "N/A"
            d_bb_stop_str = f"{d_bb_stop:.2f}" if d_bb_stop is not None else "N/A"
            d_bb_rr_info = (
                f"<br><b>日线BB盈亏比</b><br>"
                f"止盈: {d_bb_upper_str}<br>"
                f"止损: {d_bb_stop_str}<br>"
                f"盈亏比: {d_bb_rr:.2f} ({'✓' if d_bb_pass else '✗'})<br>"
            ) if d_bb_stop else ""
        
        div_info = ""
        if div_data and '60m' in div_data:
            last_feat = div_data['60m']['last_feat']
            if last_feat:
                div_info = (
                    f"<br><b>背离特征（最后一根）</b><br>"
                    f"div_bias: {last_feat.get('div_bias', 0.0):.3f}<br>"
                    f"div_last_type: {last_feat.get('div_last_type', 'none')}<br>"
                    f"div_last_age: {last_feat.get('div_last_age', 999)} bars<br>"
                    f"div_last_age_norm: {last_feat.get('div_last_age_norm', 1.0):.3f}<br>"
                    f"div_rate: {last_feat.get('div_rate', 0.0):.4f}<br>"
                )
        
        m_lz_upper_str = f"{lz_space.upper_bound:.2f}" if lz_space.upper_bound is not None else "N/A"
        m_lz_stop_str = f"{m_lz_stop:.2f}" if m_lz_stop is not None else "N/A"
        m_bb_upper_str = f"{bb_space.upper_bound:.2f}" if bb_space.upper_bound is not None else "N/A"
        m_bb_stop_str = f"{m_bb_stop:.2f}" if m_bb_stop is not None else "N/A"
        
        info_html = (
            f"<b>60m 空间分析</b><br>"
            f"<br><b>60m LZ盈亏比</b><br>"
            f"止盈: {m_lz_upper_str}<br>"
            f"止损: {m_lz_stop_str}<br>"
            f"盈亏比: {m_lz_rr:.2f} ({'✓' if m_lz_pass else '✗'})<br>"
            f"<br><b>60m BB盈亏比</b><br>"
            f"止盈: {m_bb_upper_str}<br>"
            f"止损: {m_bb_stop_str}<br>"
            f"盈亏比: {m_bb_rr:.2f} ({'✓' if m_bb_pass else '✗'})<br>"
            f"{d_lz_rr_info}"
            f"{d_bb_rr_info}"
            f"{div_info}"
            f"<br><b>通用</b><br>"
            f"最新价: {last_close:.2f}<br>"
            f"ATR(60m): {atr_60m:.2f}<br>"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
            text=info_html, align="left", showarrow=False,
            font=dict(size=11, color="#c9d1d9"),
            bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1, borderpad=8,
        )
        
        fig.update_layout(
            plot_bgcolor=TV_BG, paper_bgcolor=TV_BG,
            font=dict(color=TV_TEXT), height=800, margin=dict(l=40, r=40, t=40, b=40),
        )
        fig.update_xaxes(showgrid=True, gridcolor=TV_GRID, rangeslider_visible=False, type="category", categoryorder="array", categoryarray=x_str)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=2, col=1, rangemode="tozero")
        
        plot_divs.append(('60m', fig.to_html(full_html=False, include_plotlyjs=False)))
    
    # Tab 4-6: 背离分析 (15m, 5m, 1m)
    div_periods = ['15m', '5m', '1m']
    for div_period in div_periods:
        if not div_data or div_period not in div_data:
            continue
        
        div = div_data[div_period]
        div_df = div.get('df')
        if div_df is None or div_df.empty:
            continue
        
        if 'datetime' in div_df.columns:
            div_df_indexed = div_df.set_index('datetime')
        else:
            div_df_indexed = div_df
        
        x_str = [str(idx) for idx in div_df_indexed.index]
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
        )
        
        fig.add_trace(
            go.Candlestick(
                x=x_str,
                open=div_df_indexed["open"],
                high=div_df_indexed["high"],
                low=div_df_indexed["low"],
                close=div_df_indexed["close"],
                increasing_line_color=TV_UP,
                decreasing_line_color=TV_DOWN,
                increasing_fillcolor=TV_UP,
                decreasing_fillcolor=TV_DOWN,
                showlegend=False,
            ),
            row=1, col=1,
        )
        
        # 背离线条和标签
        pos_lines = div['pos_lines']
        neg_lines = div['neg_lines']
        pos_labels = div['pos_labels']
        neg_labels = div['neg_labels']
        
        for ln in pos_lines:
            fig.add_trace(
                go.Scatter(
                    x=ln["x"], y=ln["y"],
                    mode="lines",
                    line=dict(width=ln["width"], dash=ln["dash"]),
                    marker=dict(color=ln["color"]),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        for ln in neg_lines:
            fig.add_trace(
                go.Scatter(
                    x=ln["x"], y=ln["y"],
                    mode="lines",
                    line=dict(width=ln["width"], dash=ln["dash"]),
                    marker=dict(color=ln["color"]),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        for (x, y, text_, bg, tc) in neg_labels:
            fig.add_annotation(
                x=x, y=y, xref="x", yref="y",
                text=text_.replace("\n", "<br>"),
                showarrow=True, arrowhead=2, ax=0, ay=-25,
                bgcolor=bg, font=dict(color=tc, size=12),
                bordercolor=bg, borderwidth=1,
                row=1, col=1
            )
        
        for (x, y, text_, bg, tc) in pos_labels:
            fig.add_annotation(
                x=x, y=y, xref="x", yref="y",
                text=text_.replace("\n", "<br>"),
                showarrow=True, arrowhead=2, ax=0, ay=25,
                bgcolor=bg, font=dict(color=tc, size=12),
                bordercolor=bg, borderwidth=1,
                row=1, col=1
            )
        
        vol_colors = np.where(div_df_indexed["close"] >= div_df_indexed["open"],
                              "rgba(239,83,80,0.6)", "rgba(38,166,154,0.6)")
        fig.add_trace(
            go.Bar(x=x_str, y=div_df_indexed["volume"], marker_color=vol_colors, showlegend=False),
            row=2, col=1,
        )
        
        # 背离特征信息面板
        last_feat = div['last_feat']
        if last_feat:
            info_html = (
                f"<b>{div_period} 背离分析</b><br>"
                f"<br><b>背离特征（最后一根）</b><br>"
                f"div_bias: {last_feat.get('div_bias', 0.0):.3f}<br>"
                f"div_last_type: {last_feat.get('div_last_type', 'none')}<br>"
                f"div_last_age: {last_feat.get('div_last_age', 999)} bars<br>"
                f"div_last_age_norm: {last_feat.get('div_last_age_norm', 1.0):.3f}<br>"
                f"div_rate: {last_feat.get('div_rate', 0.0):.4f}<br>"
            )
        else:
            info_html = f"<b>{div_period} 背离分析</b><br><br>无背离信号"
        
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
            text=info_html, align="left", showarrow=False,
            font=dict(size=11, color="#c9d1d9"),
            bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1, borderpad=8,
        )
        
        fig.update_layout(
            plot_bgcolor=TV_BG, paper_bgcolor=TV_BG,
            font=dict(color=TV_TEXT), height=800, margin=dict(l=40, r=40, t=40, b=40),
        )
        fig.update_xaxes(showgrid=True, gridcolor=TV_GRID, rangeslider_visible=False, type="category", categoryorder="array", categoryarray=x_str)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=TV_GRID, row=2, col=1, rangemode="tozero")
        
        plot_divs.append((f'div_{div_period}', fig.to_html(full_html=False, include_plotlyjs=False)))
    
    period_names = {'w': '周线趋势', 'd': '日线趋势', '60m': '60m空间', 'div_15m': '15m背离', 'div_5m': '5m背离', 'div_1m': '1m背离'}
    
    tabs_html = []
    for i, (period, plot_div) in enumerate(plot_divs):
        active = 'active' if i == 0 else ''
        tabs_html.append(f'''
        <div class="tab-content {active}" id="tab-{period}">
            {plot_div}
        </div>
        ''')
    
    buttons_html = []
    for i, (period, _) in enumerate(plot_divs):
        active = 'active' if i == 0 else ''
        buttons_html.append(f'''
        <button class="tab-btn {active}" onclick="openTab(event, 'tab-{period}')">{period_names.get(period, period)}</button>
        ''')
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{display_name} 趋势分析</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            background-color: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            color: #8b949e;
        }}
        .tab-container {{
            margin-top: 20px;
        }}
        .tab-buttons {{
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }}
        .tab-btn {{
            background-color: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 14px;
            border-radius: 6px 6px 0 0;
            transition: background-color 0.2s;
        }}
        .tab-btn:hover {{
            background-color: #30363d;
        }}
        .tab-btn.active {{
            background-color: #0b0f14;
            border-bottom: 1px solid #0b0f14;
        }}
        .tab-content {{
            display: none;
            width: 100%;
        }}
        .tab-content.active {{
            display: block;
        }}
        .tab-content .plotly-graph-div {{
            width: 100% !important;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{display_name} 趋势分析</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="tab-container">
        <div class="tab-buttons">
            {''.join(buttons_html)}
        </div>
        {''.join(tabs_html)}
    </div>
    
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].classList.remove("active");
            }}
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].classList.remove("active");
            }}
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def get_stock_cache_from_db() -> dict:
    """从数据库读取股票概念缓存
    
    Returns:
        dict: {name: ts_code} 映射字典
    """
    from app.db import get_session
    from sqlalchemy import text
    
    with get_session() as session:
        sql = "SELECT name, ts_code FROM stock_concepts_cache"
        result = session.execute(text(sql))
        return {row[0]: row[1] for row in result.fetchall()}


def batch_generate_html(stock_list_path: str, stock_cache_path: str, output_dir: str):
    """批量生成股票趋势分析 HTML 文件
    
    Args:
        stock_list_path: 股票列表文件路径 (stock.xlsx) - 已废弃，现在从数据库读取
        stock_cache_path: 股票缓存文件路径 (stock_concepts_cache.xlsx) - 已废弃，从数据库读取
        output_dir: 输出目录路径
    """
    from tqdm import tqdm
    from app.stock_list_manager import get_stock_list_from_db
    
    # 从数据库读取股票列表（优先）
    stock_names = get_stock_list_from_db()
    
    # 如果数据库为空，尝试从 Excel 文件读取（向后兼容）
    if not stock_names and os.path.exists(stock_list_path):
        print(f"⚠️  数据库中没有股票数据，从 {stock_list_path} 读取")
        df_stock = pd.read_excel(stock_list_path)
        stock_names = df_stock['股票名称'].tolist()
    
    if not stock_names:
        print("❌ 错误：股票列表为空")
        return
    
    print(f"股票列表共 {len(stock_names)} 只股票")
    
    # 从数据库读取股票概念缓存
    name_to_code = get_stock_cache_from_db()
    
    # 如果数据库为空，尝试从 Excel 文件读取（向后兼容）
    if not name_to_code and os.path.exists(stock_cache_path):
        print(f"⚠️  数据库中没有股票缓存数据，从 {stock_cache_path} 读取")
        df_cache = pd.read_excel(stock_cache_path)
        name_to_code = dict(zip(df_cache['name'], df_cache['ts_code']))
    
    os.makedirs(output_dir, exist_ok=True)
    
    api = connect_pytdx()
    
    try:
        success_count = 0
        fail_count = 0
        
        for stock_name in tqdm(stock_names, desc="生成HTML"):
            try:
                if stock_name not in name_to_code:
                    print(f"\n⚠️ {stock_name}: 未找到对应代码")
                    fail_count += 1
                    continue
                
                ts_code = name_to_code[stock_name]
                symbol = ts_code.split('.')[0]
                
                output_path = os.path.join(output_dir, f"{stock_name}_趋势分析.html")
                
                kline_data = {}
                for period in ['1m', '5m', '15m', '60m', 'd', 'w']:
                    df = get_kline_data(api, symbol, period, 2000)
                    if not df.empty:
                        kline_data[period] = df
                
                if 'd' not in kline_data or kline_data['d'].empty:
                    print(f"\n⚠️ {stock_name}: 日线数据为空")
                    fail_count += 1
                    continue
                
                trend_data = {}
                for period in ['w', 'd']:
                    if period in kline_data and not kline_data[period].empty:
                        trend_result = compute_trend_for_period(kline_data[period], period)
                        if trend_result:
                            trend_data[period] = trend_result
                
                space_kline_data = {}
                for period in ['d', '60m']:
                    if period in kline_data and not kline_data[period].empty:
                        space_kline_data[period] = kline_data[period]
                
                space_data = {}
                for period in ['d', '60m']:
                    if period in space_kline_data and not space_kline_data[period].empty:
                        space_data[period] = compute_space_for_period(space_kline_data[period], period)
                        n_tail = min(255, len(space_kline_data[period]))
                        space_data[period]['df'] = space_kline_data[period].tail(n_tail).reset_index(drop=True)
                        space_data[period]['bb_upper'] = space_data[period]['bb_upper'].tail(n_tail)
                        space_data[period]['bb_mid'] = space_data[period]['bb_mid'].tail(n_tail)
                        space_data[period]['bb_lower'] = space_data[period]['bb_lower'].tail(n_tail)
                
                div_data = {}
                for period in ['1m', '5m', '15m', '60m']:
                    if period in kline_data and not kline_data[period].empty:
                        # 只使用最后 255 条数据计算背离
                        df_tail = kline_data[period].tail(255).reset_index(drop=True)
                        if 'datetime' not in df_tail.columns and 'index' in df_tail.columns:
                            df_tail = df_tail.rename(columns={'index': 'datetime'})
                        div_result = compute_divergence_for_period(df_tail, period)
                        div_data[period] = div_result
                
                vol_zscore_data = {}
                if 'd' in space_kline_data and not space_kline_data['d'].empty:
                    vol_zscore_data['d'] = compute_volume_zscore(space_kline_data['d'], win=255)
                    n_tail_d = min(255, len(space_kline_data['d']))
                    vol_zscore_data['d']['z'] = vol_zscore_data['d']['z'].tail(n_tail_d)
                
                df_w = get_kline_data(api, symbol, 'w', 500)
                if not df_w.empty:
                    vol_zscore_data['w'] = compute_volume_zscore(df_w, win=255)
                    n_tail_w = min(255, len(df_w))
                    vol_zscore_data['w']['z'] = vol_zscore_data['w']['z'].tail(n_tail_w)
                
                build_trend_tab_html(trend_data, symbol, stock_name, output_path, space_data, div_data, vol_zscore_data)
                success_count += 1
                
            except Exception as e:
                print(f"\n❌ {stock_name}: {e}")
                fail_count += 1
                continue
        
        print(f"\n✅ 批量生成完成: 成功 {success_count} 只, 失败 {fail_count} 只")
        print(f"输出目录: {output_dir}")
        
    finally:
        api.disconnect()


def main():
    parser = argparse.ArgumentParser(description="行情快照聚合脚本")
    parser.add_argument("--name", type=str, default="", help="股票名称")
    parser.add_argument("--symbol", type=str, default="", help="股票代码")
    parser.add_argument("--periods", type=str, default="1m,5m,15m,60m,d,w", help="周期列表，逗号分隔")
    parser.add_argument("--count", type=int, default=255, help="每个周期获取的 bar 数量")
    parser.add_argument("--output", type=str, default="", help="输出HTML文件路径")
    parser.add_argument("--batch", action="store_true", help="批量生成模式")
    parser.add_argument("--stock-list", type=str, default="", help="股票列表文件路径 (默认: stock.xlsx)")
    parser.add_argument("--output-dir", type=str, default="", help="批量生成输出目录 (默认: 复盘)")
    parser.add_argument("--schedule", action="store_true", help="启动定时扫描模式（每5分钟整点触发）")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    if args.schedule:
        stock_list_path = args.stock_list or os.path.join(base_dir, 'stock.xlsx')
        stock_cache_path = os.path.join(base_dir, 'stock_concepts_cache.xlsx')
        output_dir = args.output_dir or os.path.join(base_dir, '复盘')
        
        if not os.path.exists(stock_list_path):
            print(f"❌ 股票列表文件不存在: {stock_list_path}")
            return
        
        run_scheduler(stock_list_path, stock_cache_path, output_dir)
        return
    
    if args.batch:
        stock_list_path = args.stock_list or os.path.join(base_dir, 'stock.xlsx')
        stock_cache_path = os.path.join(base_dir, 'stock_concepts_cache.xlsx')
        output_dir = args.output_dir or os.path.join(base_dir, '复盘')
        
        if not os.path.exists(stock_list_path):
            print(f"❌ 股票列表文件不存在: {stock_list_path}")
            return
        
        batch_generate_html(stock_list_path, stock_cache_path, output_dir)
        return
    
    if not args.name and not args.symbol:
        parser.print_help()
        print("\n错误: 请指定 --name 或 --symbol 参数，或使用 --batch 进行批量生成")
        return
    
    api = connect_pytdx()
    
    try:
        stock_cache_path = os.path.join(base_dir, 'stock_concepts_cache.xlsx')
        
        if args.symbol:
            symbol = args.symbol
            stock_name = get_stock_name(api, symbol) or args.name
        else:
            symbol = get_stock_code_by_name(api, args.name, stock_cache_path)
            if not symbol:
                print(f"❌ 未找到股票: {args.name}")
                return
            stock_name = args.name
        
        display_name = f"{stock_name} ({symbol})" if stock_name else symbol
        print(f"\n股票: {display_name}")
        
        periods = [p.strip() for p in args.periods.split(',')]
        
        print(f"\n获取多周期行情数据（每个周期 {args.count} 条）...")
        kline_data = get_multi_period_kline(api, symbol, periods, args.count)
        
        print(f"\n获取空间分析专用数据（需要更多历史数据）...")
        space_kline_data = {}
        for period in ['d', '60m']:
            print(f"  获取 {period} 周期空间数据...", end=' ')
            df = get_kline_data(api, symbol, period, 2000)
            if not df.empty:
                print(f"✅ {len(df)} 条")
                space_kline_data[period] = df
            else:
                print("❌ 无数据")
        
        output_path = args.output
        if not output_path:
            safe_name = stock_name or symbol
            safe_name = safe_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_path = f"{safe_name}_趋势分析.html"
        
        print(f"\n计算趋势分析（周线、日线）...")
        trend_data = {}
        for period in ['w', 'd']:
            if period in kline_data and not kline_data[period].empty:
                print(f"  计算 {period} 周期趋势...", end=' ')
                trend_data[period] = compute_trend_for_period(kline_data[period], period)
                trend_data[period]['df'] = kline_data[period]
                print(f"✅ 方向={trend_data[period]['trend_dir']}, 强度={trend_data[period]['trend_strength']:.4f}")
        
        print(f"\n计算空间分析（日线、60m）...")
        space_data = {}
        for period in ['d', '60m']:
            if period in space_kline_data and not space_kline_data[period].empty:
                print(f"  计算 {period} 周期空间...", end=' ')
                space_data[period] = compute_space_for_period(space_kline_data[period], period)
                # 截取最后 255 条用于绘图
                n_tail = min(255, len(space_kline_data[period]))
                space_data[period]['df'] = space_kline_data[period].tail(n_tail).reset_index(drop=True)
                # 截取布林带数据
                space_data[period]['bb_upper'] = space_data[period]['bb_upper'].tail(n_tail)
                space_data[period]['bb_mid'] = space_data[period]['bb_mid'].tail(n_tail)
                space_data[period]['bb_lower'] = space_data[period]['bb_lower'].tail(n_tail)
                lz_sp = space_data[period]['lz_space']
                ub_str = f"{lz_sp.upper_bound:.2f}" if lz_sp.upper_bound else "N/A"
                lb_str = f"{lz_sp.lower_bound:.2f}" if lz_sp.lower_bound else "N/A"
                print(f"✅ LZ上界={ub_str}, 下界={lb_str}")
        
        print(f"\n计算背离分析（1m, 5m, 15m, 60m）...")
        div_data = {}
        div_periods = ['1m', '5m', '15m', '60m']
        for period in div_periods:
            period_key = period
            if period not in kline_data or kline_data[period].empty:
                print(f"  {period} 周期: ❌ 无数据")
                continue
            print(f"  计算 {period} 周期背离...", end=' ')
            # 只使用最后 255 条数据计算背离
            df_tail = kline_data[period].tail(255).reset_index(drop=True)
            if 'datetime' not in df_tail.columns and 'index' in df_tail.columns:
                df_tail = df_tail.rename(columns={'index': 'datetime'})
            div_result = compute_divergence_for_period(df_tail, period)
            div_data[period_key] = div_result
            last_feat = div_result['last_feat']
            if last_feat:
                print(f"✅ div_type={last_feat.get('div_last_type', 'none')}, age={last_feat.get('div_last_age', 999)}")
            else:
                print("✅ 无背离信号")
        
        print(f"\n计算成交量 ZScore（日线、周线，lookback=255）...")
        vol_zscore_data = {}
        if 'd' in space_kline_data and not space_kline_data['d'].empty:
            print(f"  计算日线 ZScore...", end=' ')
            vol_zscore_data['d'] = compute_volume_zscore(space_kline_data['d'], win=255)
            # 截取最后 255 条用于绘图
            n_tail_d = min(255, len(space_kline_data['d']))
            vol_zscore_data['d']['z'] = vol_zscore_data['d']['z'].tail(n_tail_d)
            last_z = vol_zscore_data['d']['last_z']
            print(f"✅ z={last_z:.3f}" if np.isfinite(last_z) else "✅ z=N/A")
        
        # 周线 ZScore 需要更多数据
        print(f"  获取周线空间数据...", end=' ')
        df_w = get_kline_data(api, symbol, 'w', 500)
        if not df_w.empty:
            print(f"✅ {len(df_w)} 条")
            vol_zscore_data['w'] = compute_volume_zscore(df_w, win=255)
            # 截取最后 255 条用于绘图
            n_tail_w = min(255, len(df_w))
            vol_zscore_data['w']['z'] = vol_zscore_data['w']['z'].tail(n_tail_w)
            last_z_w = vol_zscore_data['w']['last_z']
            print(f"  计算周线 ZScore... ✅ z={last_z_w:.3f}" if np.isfinite(last_z_w) else "  计算周线 ZScore... ✅ z=N/A")
        else:
            print("❌ 无数据")
        
        build_trend_tab_html(trend_data, symbol, stock_name, output_path, space_data, div_data, vol_zscore_data)
        
    finally:
        api.disconnect()
        print("\n✅ 已断开连接")


if __name__ == "__main__":
    main()


def scan_divergence_for_stock(api, symbol: str, stock_name: str) -> Dict:
    """扫描单只股票的背离情况
    
    固定扫描 1m, 5m, 15m, 60m 周期
    
    Args:
        api: pytdx API 实例
        symbol: 股票代码
        stock_name: 股票名称
    
    Returns:
        背离检测结果
    """
    periods = ['1m', '5m', '15m', '60m']
    
    results = {
        'symbol': symbol,
        'name': stock_name,
        'divergences': [],
        'has_divergence': False,
        'last_close': 0.0,  # 最新收盘价
        'pct_change': 0.0,  # 当日涨跌幅（百分比）
        'div_time': None,  # 背离发生时间
    }
    
    # 获取日线数据计算当日涨跌幅（优先，且不被覆盖）
    daily_pct_change = 0.0
    daily_close = 0.0
    try:
        df_daily = get_kline_data(api, symbol, 'd', 2)
        if not df_daily.empty and len(df_daily) >= 2:
            # 使用日线数据计算当日涨跌幅
            daily_close = float(df_daily['close'].iloc[-1])
            prev_close = float(df_daily['close'].iloc[-2])  # 昨日收盘价
            if prev_close > 0:
                daily_pct_change = ((daily_close - prev_close) / prev_close) * 100
    except Exception:
        pass  # 如果获取日线失败，使用默认值
    
    for period in periods:
        try:
            df = get_kline_data(api, symbol, period, 300)
            if df.empty or len(df) < 60:
                continue
            
            # 获取最新收盘价（仅用于背离分析，不覆盖日线数据）
            if 'close' in df.columns and not df.empty and results['last_close'] == 0.0:
                results['last_close'] = float(df['close'].iloc[-1])
            
            df_tail = df.tail(255).reset_index(drop=True)
            if 'datetime' not in df_tail.columns and 'index' in df_tail.columns:
                df_tail = df_tail.rename(columns={'index': 'datetime'})
            
            div_result = compute_divergence_for_period(df_tail, period)
            last_feat = div_result.get('last_feat')
            
            if last_feat and last_feat.get('div_last_type') != 'none':
                div_type = last_feat.get('div_last_type')
                div_age = last_feat.get('div_last_age', 999)
                div_bias = last_feat.get('div_bias', 0)
                div_time = last_feat.get('div_last_time', '')
                div_indicator = last_feat.get('div_last_indicator', '')  # 获取指标类型
                
                # 只报告最近的背离（age <= 5）
                if div_age <= 5:
                    results['divergences'].append({
                        'period': period,
                        'type': div_type,
                        'age': div_age,
                        'bias': div_bias,
                        'time': div_time,
                        'indicator': div_indicator,  # 添加指标类型
                    })
                    results['has_divergence'] = True
                    
                    # 记录背离时间（用于 5 分钟超时检查）
                    if div_time and (results['div_time'] is None or div_time > results['div_time']):
                        results['div_time'] = div_time
        
        except Exception as e:
            print(f"  ⚠️ {stock_name} {period} 周期扫描失败：{e}")
            continue
    
    # 使用日线数据覆盖（确保使用真实的当日涨跌幅）
    if daily_close > 0:
        results['last_close'] = daily_close
    if daily_pct_change != 0.0:
        results['pct_change'] = daily_pct_change
    
    return results


def scan_all_stocks(stock_list_path: str, stock_cache_path: str, output_dir: str, 
                    notify: bool = True, notified_divergences: set = None,
                    generate_files: bool = False):
    """扫描所有股票的背离情况并发送通知
    
    固定扫描 1m, 5m, 15m, 60m 周期的背离
    飞书通知包含：背离信号、盈亏比分析、日线趋势分析、合成图片
    
    去重逻辑：
    - 如果提供了 notified_divergences，只推送未推送过的背离
    - 已推送的背离信号不会被重复推送
    
    Args:
        stock_list_path: 股票列表文件路径（已废弃，现在从数据库读取）
        stock_cache_path: 股票缓存文件路径
        output_dir: 输出目录路径
        notify: 是否发送飞书通知
        notified_divergences: 已推送的背离信号集合 (symbol, period, type)
        generate_files: 是否生成 HTML/PNG 文件（默认 False，定时扫描不生成文件）
    """
    from tqdm import tqdm
    from feishu_notifier import FeishuNotifier
    from app.stock_list_manager import get_stock_list_from_db
    
    # 从数据库读取股票列表（优先）
    stock_names = get_stock_list_from_db()
    
    # 如果数据库为空，尝试从 Excel 文件读取（向后兼容）
    if not stock_names and os.path.exists(stock_list_path):
        logger.warning(f"⚠️  数据库中没有股票数据，从 {stock_list_path} 读取")
        df_stock = pd.read_excel(stock_list_path)
        stock_names = df_stock['股票名称'].tolist()
    
    if not stock_names:
        logger.error("❌ 错误：股票列表为空")
        return []
    
    # 从数据库读取股票概念缓存
    name_to_code = get_stock_cache_from_db()
    
    # 如果数据库为空，尝试从 Excel 文件读取（向后兼容）
    if not name_to_code and os.path.exists(stock_cache_path):
        logger.warning(f"⚠️  数据库中没有股票缓存数据，从 {stock_cache_path} 读取")
        df_cache = pd.read_excel(stock_cache_path)
        name_to_code = dict(zip(df_cache['name'], df_cache['ts_code']))
    
    api = connect_pytdx()
    
    all_divergences = []
    
    # 初始化 notified_divergences
    if notified_divergences is None:
        notified_divergences = set()
    
    try:
        for stock_name in tqdm(stock_names, desc="扫描背离"):
            if stock_name not in name_to_code:
                continue
            
            ts_code = name_to_code[stock_name]
            symbol = ts_code.split('.')[0]
            
            result = scan_divergence_for_stock(api, symbol, stock_name)
            
            if result['has_divergence']:
                all_divergences.append(result)
        
        if all_divergences and notify:
            notifier = FeishuNotifier()
            
            # 收集所有底背离信号（先不标记已推送）
            all_bottom_divergences = []
            
            for result in all_divergences:
                symbol = result['symbol']
                stock_name = result['name']
                last_close = result.get('last_close', 0)
                pct_change = result.get('pct_change', 0)  # 获取涨跌幅
                div_time = result.get('div_time')  # 背离时间
                
                # 检查背离时间是否超过超时限制（按周期动态调整）
                # 1m 周期：超时 6 分钟，5m 周期：超时 11 分钟，15m 周期：超时 31 分钟，60m 周期：超时 121 分钟
                if div_time and os.getenv('TEST_MODE', 'false').lower() != 'true':
                    try:
                        time_diff = datetime.now() - div_time
                        max_age_minutes = 6  # 默认 6 分钟（1m 周期）
                        
                        # 根据背离周期调整超时时间
                        for div in result['divergences']:
                            period = div.get('period', '')
                            if period == '1m':
                                max_age_minutes = 6
                            elif period == '5m':
                                max_age_minutes = 11
                            elif period == '15m':
                                max_age_minutes = 31
                            elif period == '60m':
                                max_age_minutes = 121
                        
                        if time_diff.total_seconds() > max_age_minutes * 60:
                            logger.debug(f"跳过超时信号：{symbol} 背离时间 {div_time}，距今 {time_diff.total_seconds()/60:.1f} 分钟（最大允许 {max_age_minutes} 分钟）")
                            continue
                    except Exception:
                        pass
                
                # 获取底背离信号
                for div in result['divergences']:
                    # 只收集底背离
                    if div['type'] == 'bottom':
                        # 检查是否已推送过（包含 indicator）
                        div_key = (symbol, div['period'], div['type'], div.get('indicator', ''))
                        if div_key not in notified_divergences:
                            all_bottom_divergences.append({
                                'symbol': symbol,
                                'name': stock_name,
                                'div': div,
                                'last_close': last_close,
                                'pct_change': pct_change,  # 添加涨跌幅
                            })
                        else:
                            logger.debug(f"跳过已推送信号：{symbol} {div['period']} {div['type']} [{div.get('indicator', '')}]")
            
            # 合并同一只股票的信号
            from collections import defaultdict
            stock_signals = defaultdict(list)
            for item in all_bottom_divergences:
                key = (item['symbol'], item['name'], item['last_close'], item['pct_change'])  # 添加涨跌幅到 key
                stock_signals[key].append(item['div'])
            
            # 如果有底背离，发送整合后的 Markdown 消息
            if stock_signals:
                # 构建 Markdown 消息（优化结构性和可读性）
                md_parts = []
                
                # 标题和摘要信息（使用引用块突出）
                md_parts.append("# 🎯 背离信号扫描报告")
                md_parts.append("")
                md_parts.append(f"> **📅 扫描时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                md_parts.append(f"> **📊 发现股票**: **{len(stock_signals)}** 只")
                md_parts.append("")
                md_parts.append("---")
                md_parts.append("")
                
                # 为每只股票添加背离信息（使用卡片式布局）
                for i, ((symbol, stock_name, last_close, pct_change), divs) in enumerate(stock_signals.items(), 1):
                    # 合并周期
                    periods = sorted(set(div['period'] for div in divs), 
                                   key=lambda x: {'1m':0, '5m':1, '15m':2, '30m':3, '60m':4}.get(x, 99))
                    
                    # 合并指标（去重）
                    indicators = list(set(div.get('indicator', '') for div in divs if div.get('indicator')))
                    indicator_str = f"[{'+'.join(indicators)}]" if indicators else ""
                    
                    # 获取最小的 age
                    min_age = min(div['age'] for div in divs)
                    
                    # 获取最新的时间
                    times = [div.get('time', '') for div in divs if div.get('time')]
                    latest_time = max(times) if times else ""
                    time_str = f"@{latest_time}" if latest_time else ""
                    
                    # 股价颜色（根据涨跌幅判断）
                    price_emoji = "🔺" if last_close > 0 else "🔻"
                    price_color = "red" if last_close > 0 else "green"
                    
                    # 为每个周期构建背离信息（带指标颜色）
                    period_div_info = []
                    for div in divs:
                        indicator = div.get('indicator', '')
                        div_type = div['type']
                        type_text = "底背离" if div_type == 'bottom' else "顶背离"
                        type_emoji = "🔴" if div_type == 'bottom' else "🟢"
                        
                        # 背离类型基础显示
                        base_display = f"{div['period']}{type_emoji}{type_text}"
                        
                        # 添加带颜色的指标标识
                        # MACD: 深红色 🔴, OBV: 橙色 🟠, HIST: 黄色 🟡
                        indicator_colors = {
                            'MACD': '🔴',
                            'OBV': '🟠',
                            'Hist': '🟡',
                        }
                        
                        if indicator:
                            indicator_emoji = indicator_colors.get(indicator, '⚪️')
                            period_div_info.append(f"{base_display}[{indicator_emoji}{indicator}]")
                        else:
                            period_div_info.append(base_display)
                    
                    # 合并所有周期的背离信息
                    type_display = " | ".join(period_div_info)
                    
                    # 周期可视化（带颜色等级，添加空格）
                    period_colors = {
                        '60m': '🔴', '30m': '🟠', '15m': '🟡', '5m': '🟢', '1m': '🌸',
                    }
                    period_viz = ' '.join(f"{period_colors.get(p, '⚪️')}{p}" for p in periods)
                    
                    # 股票卡片（使用标题和列表）
                    md_parts.append(f"### {i}. {stock_name} ({symbol})")
                    md_parts.append("")
                    
                    # 涨跌幅颜色和箭头
                    if pct_change > 0:
                        pct_color = "red"
                        pct_arrow = "🔺"
                    elif pct_change < 0:
                        pct_color = "green"
                        pct_arrow = "🔻"
                    else:
                        pct_color = "gray"
                        pct_arrow = "➖"
                    
                    pct_display = f"{pct_arrow} {pct_change:+.2f}%"
                    
                    md_parts.append(f"- **💰 股价**: <font color=\"{price_color}\">{last_close:.2f}</font> {price_emoji}")
                    md_parts.append(f"- **📈 涨跌幅**: <font color=\"{pct_color}\">{pct_display}</font>")
                    md_parts.append(f"- **📈 背离**: {type_display}")
                    md_parts.append(f"- **⏰ 周期**: {period_viz}")
                    md_parts.append(f"- **⏳ 年龄**: **{min_age}** 个周期")
                    if time_str:
                        md_parts.append(f"- **🕐 时间**: {time_str}")
                    md_parts.append("")
                    
                    # 添加分隔线（最后一只股票后不加）- 去掉分隔线
                    # if i < len(stock_signals):
                    #     md_parts.append("---")
                    #     md_parts.append("")
                
                # 合并并发送
                md_text = "\n".join(md_parts)
                notifier.send_markdown(md_text)
                logger.info(f"已发送背离信号扫描报告，共 {len(stock_signals)} 只股票")
                
                # 发送后统一标记为已推送
                for item in all_bottom_divergences:
                    symbol = item['symbol']
                    div = item['div']
                    div_key = (symbol, div['period'], div['type'], div.get('indicator', ''))
                    notified_divergences.add(div_key)
            else:
                logger.debug("未发现未推送的底背离信号")
                    
        
        return all_divergences
    
    finally:
        api.disconnect()


def is_trading_time() -> bool:
    """判断当前是否在 A 股交易时段
    
    A 股交易时间：
    - 上午：9:25 - 11:30 (包含集合竞价)
    - 下午：13:00 - 15:00
    
    Returns:
        是否在交易时段
    """
    now = datetime.now()
    current_time = now.time()
    
    # 上午交易时段 (9:25 开始包含集合竞价)
    morning_start = datetime.strptime("09:25", "%H:%M").time()
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    
    # 下午交易时段
    afternoon_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_end = datetime.strptime("15:00", "%H:%M").time()
    
    is_morning = morning_start <= current_time <= morning_end
    is_afternoon = afternoon_start <= current_time <= afternoon_end
    
    return is_morning or is_afternoon


def is_trading_day() -> bool:
    """判断今天是否是交易日
    
    使用 qstock 库获取最新交易日期，与当前日期比较
    
    Returns:
        是否是交易日
    """
    try:
        import qstock as qs
        latest_date = qs.latest_trade_date()  # 返回格式: '2026-03-03'
        today = datetime.now().strftime('%Y-%m-%d')
        return latest_date == today
    except Exception:
        # 如果 qstock 获取失败，使用简单的周末判断
        now = datetime.now()
        return now.weekday() < 5  # 0-4 为周一到周五


def run_scheduler(stock_list_path: str, stock_cache_path: str, output_dir: str, 
                  interval_minutes: int = 5):
    """运行定时扫描任务
    
    每 interval_minutes 分钟整点触发扫描（仅在交易日和交易时段）
    
    飞书通知内容：
    - 背离信号：扫描 1m, 5m, 15m, 60m 周期
    - 盈亏比分析：60m 和日线数据
    - 日线趋势分析：日线数据
    - 合成图片：60m, 15m, 5m, 1m 四个周期 K 线图
    
    Args:
        stock_list_path: 股票列表文件路径
        stock_cache_path: 股票缓存文件路径
        output_dir: 输出目录路径
        interval_minutes: 扫描间隔（分钟）
    """
    import time
    
    print(f"启动定时扫描任务，每 {interval_minutes} 分钟整点触发")
    print(f"股票列表: {stock_list_path}")
    print(f"输出目录: {output_dir}")
    print(f"交易时段: 09:30-11:30, 13:00-15:00")
    print("-" * 50)
    
    def job():
        now = datetime.now()
        
        # 检查是否是交易日
        if not is_trading_day():
            print(f"⏸️ [{now.strftime('%Y-%m-%d %H:%M:%S')}] 今天不是交易日，跳过扫描")
            return
        
        # 检查是否在交易时段
        if not is_trading_time():
            print(f"⏸️ [{now.strftime('%Y-%m-%d %H:%M:%S')}] 当前不在交易时段，跳过扫描")
            return
        
        print(f"\n⏰ [{now.strftime('%Y-%m-%d %H:%M:%S')}] 开始扫描...")
        
        try:
            divergences = scan_all_stocks(
                stock_list_path, 
                stock_cache_path, 
                output_dir,
                notify=True,
                notified_divergences=notified_divergences,
            )
            
            if not divergences:
                print("✅ 本次扫描未发现背离信号")
        
        except Exception as e:
            print(f"❌ 扫描失败: {e}")
    
    # 计算距离下一个整点时间的秒数
    def get_seconds_to_next_trigger():
        now = datetime.now()
        current_minute = now.minute
        current_second = now.second
        
        # 计算下一个触发点（整点）
        next_trigger_minute = ((current_minute // interval_minutes) + 1) * interval_minutes
        
        # 如果超过60分钟，需要进位到下一小时
        if next_trigger_minute >= 60:
            next_trigger_minute = 0
            seconds_to_next_hour = (60 - current_minute - 1) * 60 + (60 - current_second)
            return seconds_to_next_hour + next_trigger_minute * 60
        else:
            return (next_trigger_minute - current_minute) * 60 - current_second
    
    # 首次扫描（如果在交易时段）
    print("检查是否需要首次扫描...")
    if is_trading_day() and is_trading_time():
        job()
    else:
        print("⏸️ 当前不在交易时段，等待下一个触发点...")
    
    # 进入循环
    print(f"\n进入定时循环，等待下一个触发点...")
    while True:
        seconds_to_wait = get_seconds_to_next_trigger()
        
        if seconds_to_wait > 0:
            # 等待到下一个触发点
            next_trigger_time = datetime.now() + timedelta(seconds=seconds_to_wait)
            print(f"下次触发时间: {next_trigger_time.strftime('%Y-%m-%d %H:%M:%S')} ({seconds_to_wait}秒后)")
            time.sleep(seconds_to_wait)
        
        # 执行任务
        job()
        
        # 等待一小段时间避免重复触发
        time.sleep(1)
