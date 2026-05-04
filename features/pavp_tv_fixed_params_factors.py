# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
except Exception as exc:
    raise RuntimeError('请先安装 pytdx: pip install pytdx') from exc

# =========================
# 固定为截图参数
# =========================
PIVOT_LENGTH = 20
PROFILE_LEVELS = 25
VALUE_AREA_PCT = 0.68
PROFILE_PLACEMENT = 'Left'
PROFILE_WIDTH = 0.30
SHOW_VOLUME_PROFILE = True
SHOW_POC = True
SHOW_VAH = True
SHOW_VAL = True
SHOW_VA_BG = True
SHOW_BG_FILL = True
SHOW_LEVEL_LABELS = True
VWCB = True

SERVERS = [
    ('119.147.212.81', 7709), ('119.147.164.60', 7709), ('14.215.128.18', 7709),
    ('14.215.128.116', 7709), ('101.133.156.38', 7709), ('114.80.149.19', 7709),
    ('115.238.90.165', 7709), ('123.125.108.23', 7709), ('180.153.18.170', 7709),
    ('202.108.253.131', 7709),
]


def normalize_freq(freq: str) -> str:
    f = str(freq).strip().lower()
    if f in {'d', '1d', 'day', 'daily', '101'}:
        return 'd'
    if f in {'w', '1w', 'week', 'weekly'}:
        return 'w'
    if f in {'m', 'mo', 'month', 'monthly'}:
        return 'mo'
    if f in {'60', '60m', '1h'}:
        return '60m'
    if f in {'30', '30m'}:
        return '30m'
    if f in {'15', '15m'}:
        return '15m'
    if f in {'5', '5m'}:
        return '5m'
    if f in {'1', '1m'}:
        return '1m'
    raise ValueError(f'不支持的频率: {freq}')


def _category_from_freq(freq: str) -> int:
    f = normalize_freq(freq)
    return {
        'd': TDXParams.KLINE_TYPE_RI_K,
        'w': TDXParams.KLINE_TYPE_WEEKLY,
        'mo': TDXParams.KLINE_TYPE_MONTHLY,
        '60m': TDXParams.KLINE_TYPE_1HOUR,
        '30m': TDXParams.KLINE_TYPE_30MIN,
        '15m': TDXParams.KLINE_TYPE_15MIN,
        '5m': TDXParams.KLINE_TYPE_5MIN,
        '1m': TDXParams.KLINE_TYPE_1MIN,
    }[f]


def _market_from_symbol(symbol: str) -> int:
    return 1 if str(symbol).startswith(('5', '6', '9')) else 0


def connect_pytdx() -> TdxHq_API:
    errors: List[str] = []
    for host, port in SERVERS:
        try:
            api = TdxHq_API(raise_exception=True, auto_retry=True)
            if api.connect(host, port):
                return api
        except Exception as exc:
            errors.append(f'{host}:{port} {exc}')
    raise RuntimeError('pytdx 连接失败: ' + '; '.join(errors[-5:]))


def fetch_kline_pytdx(symbol: str, freq: str, count: int) -> pd.DataFrame:
    api = connect_pytdx()
    try:
        cat = _category_from_freq(freq)
        mkt = _market_from_symbol(symbol)
        size = 800
        frames: List[pd.DataFrame] = []
        start = 0
        target = max(int(count), 300)
        while start < target + size:
            recs = api.get_security_bars(cat, mkt, symbol, start, size)
            if not recs:
                break
            d = pd.DataFrame(recs)
            if 'datetime' in d.columns:
                d['datetime'] = pd.to_datetime(d['datetime']).dt.tz_localize(None)
            else:
                d['datetime'] = pd.to_datetime(d[['year', 'month', 'day', 'hour', 'minute']].astype(int))
            if 'vol' in d.columns:
                d = d.rename(columns={'vol': 'volume'})
            if 'amount' not in d.columns:
                d['amount'] = np.nan
            keep = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
            frames.append(d[keep].sort_values('datetime'))
            if len(recs) < size:
                break
            start += size
        if not frames:
            raise RuntimeError('pytdx 无数据')
        out = (pd.concat(frames).sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last').tail(count).set_index('datetime'))
        return out.astype(float)
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def tv_pivots_confirmed(high: np.ndarray, low: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    pvt_high = np.full(n, np.nan)
    pvt_low = np.full(n, np.nan)
    win = 2 * length + 1
    for i in range(win - 1, n):
        c = i - length
        lo = c - length
        hi = c + length + 1
        hwin = high[lo:hi]
        lwin = low[lo:hi]
        if np.all(np.isfinite(hwin)) and np.isfinite(high[c]) and high[c] == np.max(hwin):
            pvt_high[i] = high[c]
        if np.all(np.isfinite(lwin)) and np.isfinite(low[c]) and low[c] == np.min(lwin):
            pvt_low[i] = low[c]
    return pvt_high, pvt_low


def _weighted_skewness(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() < 2:
        return np.nan
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    ws = w.sum()
    mean = np.sum(w * v) / ws
    var = np.sum(w * (v - mean) ** 2) / ws
    if var <= 0:
        return 0.0
    std = np.sqrt(var)
    third = np.sum(w * (v - mean) ** 3) / ws
    return float(third / (std ** 3))


@dataclass
class ProfileSegment:
    kind: str  # fixed / developing
    confirm_index: int
    start_index: int
    end_index: int
    price_highest: float
    price_lowest: float
    price_step: float
    hist: np.ndarray
    hist_sum: float
    poc_level: int
    level_above_poc: int
    level_below_poc: int
    poc_price: float
    vah_price: float
    val_price: float
    traded_volume_window: float


def _calc_value_area(hist: np.ndarray, profile_levels: int, value_area_pct: float) -> Tuple[int, int, int]:
    poc_level = int(np.argmax(hist))
    total_volume_traded = float(np.sum(hist) * value_area_pct)
    value_area = float(hist[poc_level])
    level_above_poc = poc_level
    level_below_poc = poc_level
    while value_area < total_volume_traded:
        if level_below_poc == 0 and level_above_poc == profile_levels - 1:
            break
        volume_above_poc = float(hist[level_above_poc + 1]) if level_above_poc < profile_levels - 1 else 0.0
        volume_below_poc = float(hist[level_below_poc - 1]) if level_below_poc > 0 else 0.0
        if volume_below_poc == 0 and volume_above_poc == 0:
            break
        if volume_above_poc >= volume_below_poc:
            value_area += volume_above_poc
            level_above_poc += 1
        else:
            value_area += volume_below_poc
            level_below_poc -= 1
    return poc_level, level_above_poc, level_below_poc


def _build_histogram_by_offsets(high_arr, low_arr, vol_arr, current_idx, offsets, price_lowest, price_highest, profile_levels):
    hist = np.zeros(profile_levels + 1, dtype=float)
    price_step = (price_highest - price_lowest) / float(profile_levels)
    if not np.isfinite(price_step) or price_step <= 0:
        return hist, np.nan
    for off in offsets:
        j = current_idx - off
        if j < 0 or j >= len(high_arr):
            continue
        bar_h = float(high_arr[j])
        bar_l = float(low_arr[j])
        bar_v = float(vol_arr[j])
        if not np.isfinite(bar_h) or not np.isfinite(bar_l) or not np.isfinite(bar_v) or bar_v <= 0:
            continue
        span = bar_h - bar_l
        for level in range(profile_levels + 1):
            price_level = price_lowest + level * price_step
            if bar_h >= price_level and bar_l < price_level + price_step:
                alloc = bar_v if span == 0 else bar_v * (price_step / span)
                hist[level] += alloc
    return hist, price_step


def _historical_fixed_segment(h, l, v, i, profile_length, offset, profile_levels, value_area_pct) -> Optional[ProfileSegment]:
    # 对齐 Pine f_getHighLow(profileLength, proceed, pvtLength)
    # high/low 范围: offset .. offset+profileLength (共 profileLength+1 根)
    price_highest = float(h[i - offset])
    price_lowest = float(l[i - offset])
    traded_volume = 0.0
    for x in range(profile_length):
        j = i - (offset + x)
        if j < 0:
            return None
        price_highest = max(price_highest, float(h[j]))
        price_lowest = min(price_lowest, float(l[j]))
        vv = float(v[j])
        if np.isfinite(vv):
            traded_volume += vv
    j_extra = i - (offset + profile_length)
    if j_extra < 0:
        return None
    price_highest = max(price_highest, float(h[j_extra]))
    price_lowest = min(price_lowest, float(l[j_extra]))
    price_step = (price_highest - price_lowest) / float(profile_levels)
    nz_volume = 0.0 if not np.isfinite(v[i]) else float(v[i])
    if not (nz_volume > 0 and profile_length > 0 and price_step > 0 and i > profile_length):
        return None

    # 对齐 Pine: barIndex = 1..profileLength, real offset = barIndex + pvtLength
    offsets = [x + offset for x in range(1, profile_length + 1)]
    hist, price_step = _build_histogram_by_offsets(h, l, v, i, offsets, price_lowest, price_highest, profile_levels)
    hist_sum = float(hist.sum())
    if hist_sum <= 0:
        return None
    poc_level, level_above_poc, level_below_poc = _calc_value_area(hist, profile_levels, value_area_pct)
    poc_price = float(price_lowest + (poc_level + 0.50) * price_step)
    vah_price = float(price_lowest + (level_above_poc + 1.00) * price_step)
    val_price = float(price_lowest + (level_below_poc + 0.00) * price_step)
    # 左右边界也按 Pine 画法：bar_index[pvtLength]-profileLength 到 bar_index[pvtLength]
    end_index = i - offset
    start_index = end_index - profile_length
    return ProfileSegment('fixed', i, start_index, end_index, price_highest, price_lowest, price_step, hist, hist_sum,
                          poc_level, level_above_poc, level_below_poc, poc_price, vah_price, val_price, traded_volume)


def _developing_segment(h, l, v, pvt_high_confirm, pvt_low_confirm, i, pivot_length, profile_levels, value_area_pct) -> Optional[ProfileSegment]:
    x2 = 0
    for k in range(i + 1):
        if np.isfinite(pvt_high_confirm[k]) or np.isfinite(pvt_low_confirm[k]):
            x2 = k
    profile_length = i - x2 + pivot_length
    start_index = i - profile_length
    end_index = i
    if start_index < 0 or profile_length <= 0:
        return None
    price_highest = float(np.max(h[start_index:i + 1]))
    price_lowest = float(np.min(l[start_index:i + 1]))
    price_step = (price_highest - price_lowest) / float(profile_levels)
    nz_volume = 0.0 if not np.isfinite(v[i]) else float(v[i])
    if not (nz_volume > 0 and price_step > 0):
        return None
    offsets = list(range(1, profile_length + 1))
    hist, price_step = _build_histogram_by_offsets(h, l, v, i, offsets, price_lowest, price_highest, profile_levels)
    hist_sum = float(hist.sum())
    if hist_sum <= 0:
        return None
    poc_level, level_above_poc, level_below_poc = _calc_value_area(hist, profile_levels, value_area_pct)
    poc_price = float(price_lowest + (poc_level + 0.50) * price_step)
    vah_price = float(price_lowest + (level_above_poc + 1.00) * price_step)
    val_price = float(price_lowest + (level_below_poc + 0.00) * price_step)
    traded_volume = float(np.nansum(v[start_index:i + 1]))
    return ProfileSegment('developing', i, start_index, end_index, price_highest, price_lowest, price_step, hist, hist_sum,
                          poc_level, level_above_poc, level_below_poc, poc_price, vah_price, val_price, traded_volume)


def compute_pavp(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[ProfileSegment], Optional[ProfileSegment]]:
    out = df.copy()
    h = out['high'].to_numpy(dtype=float)
    l = out['low'].to_numpy(dtype=float)
    c = out['close'].to_numpy(dtype=float)
    v = out['volume'].to_numpy(dtype=float)
    n = len(out)

    pvt_high_confirm, pvt_low_confirm = tv_pivots_confirmed(h, l, PIVOT_LENGTH)
    out['pvt_high_confirm'] = pvt_high_confirm
    out['pvt_low_confirm'] = pvt_low_confirm

    factor_cols = [
        'segment_kind', 'segment_confirm_index', 'segment_start_index', 'segment_end_index',
        'profile_high', 'profile_low', 'profile_length_tv', 'hist_total_volume',
        'poc_price', 'vah_price', 'val_price',
        'va_width', 'va_width_pct_of_profile', 'va_pos_01', 'poc_in_va_pos_01',
        'poc_pos_01_in_profile', 'close_to_poc_pct',
        'profile_skewness', 'profile_concentration_hhi', 'profile_poc_share',
        'profile_balance_ratio', 'profile_upper_tail_share', 'profile_lower_tail_share'
    ]
    for col in factor_cols:
        if col == 'segment_kind':
            out[col] = None
        else:
            out[col] = np.nan

    def _write_segment(idx: int, seg: ProfileSegment, kind_override: Optional[str] = None) -> None:
        kind = seg.kind if kind_override is None else kind_override
        va_width = seg.vah_price - seg.val_price
        profile_width = seg.price_highest - seg.price_lowest
        mids = seg.price_lowest + (np.arange(PROFILE_LEVELS + 1, dtype=float) + 0.5) * seg.price_step
        shares = seg.hist / seg.hist_sum
        poc_share = float(np.max(seg.hist) / seg.hist_sum)
        upper_tail_share = float(np.sum(shares[seg.level_above_poc + 1:])) if seg.level_above_poc + 1 < len(shares) else 0.0
        lower_tail_share = float(np.sum(shares[:seg.level_below_poc])) if seg.level_below_poc > 0 else 0.0
        balance_ratio = np.nan
        if lower_tail_share > 0:
            balance_ratio = upper_tail_share / lower_tail_share
        elif upper_tail_share == 0:
            balance_ratio = 1.0

        out.iat[idx, out.columns.get_loc('segment_kind')] = kind
        out.iat[idx, out.columns.get_loc('segment_confirm_index')] = float(seg.confirm_index)
        out.iat[idx, out.columns.get_loc('segment_start_index')] = float(seg.start_index)
        out.iat[idx, out.columns.get_loc('segment_end_index')] = float(seg.end_index)
        out.iat[idx, out.columns.get_loc('profile_high')] = seg.price_highest
        out.iat[idx, out.columns.get_loc('profile_low')] = seg.price_lowest
        out.iat[idx, out.columns.get_loc('profile_length_tv')] = float(seg.end_index - seg.start_index)
        out.iat[idx, out.columns.get_loc('hist_total_volume')] = seg.hist_sum
        out.iat[idx, out.columns.get_loc('poc_price')] = seg.poc_price
        out.iat[idx, out.columns.get_loc('vah_price')] = seg.vah_price
        out.iat[idx, out.columns.get_loc('val_price')] = seg.val_price
        out.iat[idx, out.columns.get_loc('va_width')] = va_width
        out.iat[idx, out.columns.get_loc('va_width_pct_of_profile')] = np.nan if profile_width == 0 else float(va_width / profile_width)
        out.iat[idx, out.columns.get_loc('va_pos_01')] = np.nan if va_width == 0 else float((c[idx] - seg.val_price) / va_width)
        out.iat[idx, out.columns.get_loc('poc_in_va_pos_01')] = np.nan if va_width == 0 else float((seg.poc_price - seg.val_price) / va_width)
        out.iat[idx, out.columns.get_loc('poc_pos_01_in_profile')] = np.nan if profile_width == 0 else float((seg.poc_price - seg.price_lowest) / profile_width)
        out.iat[idx, out.columns.get_loc('close_to_poc_pct')] = np.nan if seg.poc_price == 0 else float((c[idx] - seg.poc_price) / seg.poc_price)
        out.iat[idx, out.columns.get_loc('profile_skewness')] = _weighted_skewness(mids, seg.hist)
        out.iat[idx, out.columns.get_loc('profile_concentration_hhi')] = float(np.sum(shares ** 2))
        out.iat[idx, out.columns.get_loc('profile_poc_share')] = poc_share
        out.iat[idx, out.columns.get_loc('profile_balance_ratio')] = balance_ratio
        out.iat[idx, out.columns.get_loc('profile_upper_tail_share')] = upper_tail_share
        out.iat[idx, out.columns.get_loc('profile_lower_tail_share')] = lower_tail_share

    fixed_segments: List[ProfileSegment] = []
    x1 = 0
    x2 = 0
    last_dev: Optional[ProfileSegment] = None
    fixed_event_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        proceed = bool(np.isfinite(pvt_high_confirm[i]) or np.isfinite(pvt_low_confirm[i]))
        if proceed:
            x1 = x2
            x2 = i

        profile_length = x2 - x1
        if proceed:
            seg = _historical_fixed_segment(h, l, v, i, profile_length, PIVOT_LENGTH, PROFILE_LEVELS, VALUE_AREA_PCT)
            if seg is not None:
                fixed_segments.append(seg)
                fixed_event_mask[i] = True

        # 为副图生成逐 bar 连续的 developing 因子序列
        dev_i = _developing_segment(h, l, v, pvt_high_confirm, pvt_low_confirm, i, PIVOT_LENGTH, PROFILE_LEVELS, VALUE_AREA_PCT)
        if dev_i is not None:
            _write_segment(i, dev_i, kind_override='developing')
            last_dev = dev_i

    # 用独立标记列保留 fixed 段发生点，避免被 developing 序列覆盖
    out['fixed_event'] = fixed_event_mask.astype(int)
    out['developing_event'] = out['segment_kind'].eq('developing').astype(int)
    return out, fixed_segments, last_dev


def _rgba(hex_color: str, alpha: float) -> str:
    s = hex_color.lstrip('#')
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _volume_weighted_bar_colors(df: pd.DataFrame, sma_len: int = 89, up_thresh: float = 1.618, low_thresh: float = 0.618) -> List[str]:
    vol = df['volume'].astype(float)
    v_sma = vol.rolling(sma_len, min_periods=1).mean()
    colors: List[str] = []
    for o, cl, vv, ma in zip(df['open'], df['close'], vol, v_sma):
        bull = cl > o
        if vv > ma * up_thresh:
            colors.append('#006400' if bull else '#910000')
        elif vv < ma * low_thresh:
            colors.append('#7FFFD4' if bull else '#FF9800')
        else:
            colors.append('#00c853' if bull else '#ff5252')
    return colors


def build_html(df_full: pd.DataFrame, df_plot: pd.DataFrame, fixed_segments: List[ProfileSegment], dev: Optional[ProfileSegment], out_html: str, title: str):
    if go is None or make_subplots is None:
        raise RuntimeError('未安装 plotly')
    plot_start = len(df_full) - len(df_plot)
    x_num = np.arange(len(df_plot), dtype=float)
    intraday = len(df_plot.index) > 1 and (df_plot.index[1] - df_plot.index[0]) < pd.Timedelta('20H')
    tick_text = [ts.strftime('%Y-%m-%d %H:%M') if intraday else ts.strftime('%Y-%m-%d') for ts in df_plot.index]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.60, 0.20, 0.20],
                        subplot_titles=(title, 'PAVP Value Area Position', 'PAVP Factors'))

    colors = _volume_weighted_bar_colors(df_plot)
    for i, (o, hi, lo, cl, clr) in enumerate(zip(df_plot['open'], df_plot['high'], df_plot['low'], df_plot['close'], colors)):
        fig.add_trace(go.Scatter(x=[i, i], y=[lo, hi], mode='lines', line=dict(color=clr, width=1), hoverinfo='skip', showlegend=False), row=1, col=1)
        body_low = min(o, cl)
        body_high = max(o, cl)
        fig.add_shape(type='rect', x0=i - 0.32, x1=i + 0.32, y0=body_low, y1=(body_high if body_high > body_low else body_low + 1e-9),
                      line=dict(color=clr, width=1), fillcolor=clr, xref='x1', yref='y1')

    def draw_segment(seg: ProfileSegment, draw_boxes: bool, line_width: int = 2):
        left = seg.start_index - plot_start
        right = seg.end_index - plot_start
        if right < 0 or left > len(df_plot) - 1:
            return
        left_clip = max(left, 0)
        right_clip = min(right, len(df_plot) - 1)
        if SHOW_BG_FILL:
            fig.add_shape(type='rect', x0=left_clip, x1=right_clip, y0=seg.price_lowest, y1=seg.price_highest,
                          line=dict(color=_rgba('#2962ff', 0.18), width=1, dash='dot'), fillcolor=_rgba('#2962ff', 0.05), xref='x1', yref='y1')
        if SHOW_VA_BG:
            fig.add_shape(type='rect', x0=left_clip, x1=right_clip, y0=seg.val_price, y1=seg.vah_price,
                          line=dict(color='rgba(0,0,0,0)', width=0), fillcolor=_rgba('#00134d', 0.18), xref='x1', yref='y1')
        if draw_boxes and SHOW_VOLUME_PROFILE:
            max_hist = float(np.max(seg.hist)) if np.max(seg.hist) > 0 else 1.0
            for level in range(PROFILE_LEVELS):
                share = float(seg.hist[level] / max_hist)
                width = int(share * (seg.end_index - seg.start_index) * PROFILE_WIDTH)
                if PROFILE_PLACEMENT.lower() == 'right':
                    x0 = right_clip - width
                    x1 = right_clip
                else:
                    x0 = left
                    x1 = x0 + width
                x0 = max(x0, 0)
                x1 = min(x1, len(df_plot) - 1)
                if x1 <= x0:
                    continue
                y0 = seg.price_lowest + (level + 0.10) * seg.price_step
                y1 = seg.price_lowest + (level + 0.90) * seg.price_step
                in_va = seg.level_below_poc <= level <= seg.level_above_poc
                fill = _rgba('#fbc02d', 0.32) if in_va else _rgba('#434651', 0.32)
                line = _rgba('#fbc02d', 0.45) if in_va else _rgba('#434651', 0.45)
                fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color=line, width=1), fillcolor=fill, xref='x1', yref='y1')
        if SHOW_POC:
            fig.add_trace(go.Scatter(x=[left_clip, right_clip], y=[seg.poc_price, seg.poc_price], mode='lines', line=dict(color='#ff0000', width=line_width), name='POC', showlegend=False), row=1, col=1)
        if SHOW_VAH:
            fig.add_trace(go.Scatter(x=[left_clip, right_clip], y=[seg.vah_price, seg.vah_price], mode='lines', line=dict(color='#2962ff', width=line_width), name='VAH', showlegend=False), row=1, col=1)
        if SHOW_VAL:
            fig.add_trace(go.Scatter(x=[left_clip, right_clip], y=[seg.val_price, seg.val_price], mode='lines', line=dict(color='#2962ff', width=line_width), name='VAL', showlegend=False), row=1, col=1)
        if SHOW_POC:
            fig.add_shape(type='rect', x0=left_clip, x1=right_clip,
                          y0=seg.price_lowest + (seg.poc_level + 0.40) * seg.price_step,
                          y1=seg.price_lowest + (seg.poc_level + 0.60) * seg.price_step,
                          line=dict(color='#ff0000', width=1), fillcolor=_rgba('#ff0000', 0.35), xref='x1', yref='y1')

    # 历史固定 profile 都画出来
    for seg in fixed_segments:
        draw_segment(seg, draw_boxes=False, line_width=2)

    # 最后一段 developing profile 再额外画 profile box，贴近 TV 最右侧效果
    if dev is not None:
        draw_segment(dev, draw_boxes=True, line_width=2)

    # pivot labels
    for j in range(len(df_plot)):
        full_j = plot_start + j
        if np.isfinite(df_full['pvt_high_confirm'].iloc[full_j]):
            anchor = full_j - PIVOT_LENGTH - plot_start
            if 0 <= anchor < len(df_plot):
                fig.add_annotation(x=anchor, y=float(df_full['pvt_high_confirm'].iloc[full_j]), text=f"{float(df_full['pvt_high_confirm'].iloc[full_j]):.2f}",
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='#d4a5a5', ax=0, ay=-22,
                                   bgcolor='#d4a5a5', font=dict(color='white', size=10), xref='x1', yref='y1')
        if np.isfinite(df_full['pvt_low_confirm'].iloc[full_j]):
            anchor = full_j - PIVOT_LENGTH - plot_start
            if 0 <= anchor < len(df_plot):
                fig.add_annotation(x=anchor, y=float(df_full['pvt_low_confirm'].iloc[full_j]), text=f"{float(df_full['pvt_low_confirm'].iloc[full_j]):.2f}",
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='#8faadc', ax=0, ay=22,
                                   bgcolor='#8faadc', font=dict(color='white', size=10), xref='x1', yref='y1')

    fig.add_trace(go.Scatter(x=x_num, y=df_plot['va_pos_01'], mode='lines', name='va_pos_01'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df_plot['poc_in_va_pos_01'], mode='lines', name='poc_in_va_pos_01'), row=2, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash='dot', line_color='#90a4ae', row=2, col=1)
    fig.add_hline(y=1.0, line_width=1, line_dash='dot', line_color='#90a4ae', row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df_plot['profile_skewness'], mode='lines', name='profile_skewness'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df_plot['profile_concentration_hhi'], mode='lines', name='profile_concentration_hhi'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df_plot['profile_poc_share'], mode='lines', name='profile_poc_share'), row=3, col=1)

    tick_step = max(1, len(df_plot) // 10)
    tickvals = list(range(0, len(df_plot), tick_step))
    if tickvals and tickvals[-1] != len(df_plot) - 1:
        tickvals.append(len(df_plot) - 1)
    ticktext = [tick_text[i] for i in tickvals] if tickvals else []

    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, hovermode='x unified',
                      margin=dict(l=40, r=20, t=80, b=40), height=1200)
    for r in [1, 2, 3]:
        fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)
    fig.write_html(out_html, include_plotlyjs='cdn')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='固定参数版 TV-Pine PAVP 复刻脚本')
    p.add_argument('--symbol', required=True)
    p.add_argument('--freq', default='d')
    p.add_argument('--bars', type=int, default=300)
    p.add_argument('--fetch-bars', type=int, default=1200)
    p.add_argument('--out-csv', default='pavp_tv_fixed.csv')
    p.add_argument('--out-html', default='pavp_tv_fixed.html')
    return p


def main():
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)
    fetch_bars = max(args.fetch_bars, args.bars, 2 * PIVOT_LENGTH + 200)
    raw = fetch_kline_pytdx(args.symbol, args.freq, fetch_bars)
    out_full, fixed_segments, dev = compute_pavp(raw)
    out_plot = out_full.tail(args.bars).copy()
    out_plot.to_csv(args.out_csv, encoding='utf-8-sig')
    title = f'{args.symbol} [{args.freq}] TV Pine PAVP Replicated (fixed params)'
    build_html(out_full, out_plot, fixed_segments, dev, args.out_html, title)
    print(f'CSV 已生成: {args.out_csv}')
    print(f'HTML 已生成: {args.out_html}')


if __name__ == '__main__':
    main()
