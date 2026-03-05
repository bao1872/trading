# -*- coding: utf-8 -*-
"""
获取逐笔交易数据并分析主买主卖（30天汇总版）

功能：
1. 获取指定股票最近30天的逐笔交易数据
2. 按天汇总主买/主卖金额和笔数
3. 绘制K线图 + 3个副图（主买主卖金额、主买主卖笔数、MACD）
4. 输出HTML

Usage:
    # 批量处理（自动输出 HTML 和 PNG）
    python app/tick_analysis_30d.py --batch stock.xlsx --output-dir 复盘 --days 30

    # 单只股票（自动输出 HTML 和 PNG）
    python app/tick_analysis_30d.py --symbol 000426 --days 30 --output 000426_tick_analysis.html
"""
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import json

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.pytdx_client import (
    connect_pytdx,
    get_stock_name,
    get_daily_bars,
    get_tick_data_for_date,
    market_from_code,
    TdxHq_API,
)



def get_30day_tick_summary(api: TdxHq_API, symbol: str, days: int = 30) -> pd.DataFrame:
    """获取最近 N 天的逐笔数据汇总"""
    end_date = datetime.now()
    
    # 获取日线数据（多获取 255 天用于 ZScore 计算）
    daily_df = get_daily_bars(api, symbol, days + 255)
    
    if daily_df.empty:
        print("❌ 未获取到日线数据")
        return pd.DataFrame()
    
    print(f"\n正在获取 {symbol} 日线数据...")
    
    results = []
    total = len(daily_df)
    
    # 多获取几天的日线数据，以确保有足够的交易日数据
    tick_start_idx = max(0, len(daily_df) - days - 5)
    
    for idx, row in daily_df.iterrows():
        date_str = row['datetime'].strftime('%Y-%m-%d')
        date_int = int(row['datetime'].strftime('%Y%m%d'))
        
        # 只获取最后N天的逐笔数据
        if idx >= tick_start_idx:
            print(f"  [{idx - tick_start_idx + 1}/{len(daily_df) - tick_start_idx}] 获取 {date_str} ...", end=' ')
            
            tick_data = get_tick_data_for_date(api, symbol, date_int)
            
            if tick_data:
                tick_data['date'] = row['datetime']
                tick_data['open'] = row['open']
                tick_data['high'] = row['high']
                tick_data['low'] = row['low']
                tick_data['close'] = row['close']
                tick_data['volume'] = row['volume']
                results.append(tick_data)
                print(f"✅ {tick_data['total_trades']} 笔")
            else:
                print("❌ 无数据")
        else:
            # 前面的数据只保留日线，用于布林带计算
            results.append({
                'date': row['datetime'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'buy_trades': 0,
                'sell_trades': 0,
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_amount': 0,
                'sell_amount': 0,
                'total_trades': 0,
            })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 计算比值
    df['buy_sell_trade_ratio'] = df['buy_trades'] / df['sell_trades'].replace(0, np.nan)
    df['buy_sell_amount_ratio'] = df['buy_amount'] / df['sell_amount'].replace(0, np.nan)
    df['net_buy_amount'] = df['buy_amount'] - df['sell_amount']
    df['net_buy_volume'] = df['buy_volume'] - df['sell_volume']
    
    return df


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """计算MACD指标"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return {
        'dif': dif,
        'dea': dea,
        'macd': macd,
    }


def compute_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """计算布林带指标"""
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return {
        'boll_mid': mid,
        'boll_upper': upper,
        'boll_lower': lower,
    }


def compute_volume_zscore(volume: pd.Series, win: int = 255) -> pd.Series:
    """计算成交量ZScore指标（默认255日窗口）"""
    mu = volume.rolling(win, min_periods=win).mean()
    sd = volume.rolling(win, min_periods=win).std(ddof=0)
    z = (volume - mu) / sd.replace(0.0, np.nan)
    return z


def build_html(df: pd.DataFrame, symbol: str, output_path: str, display_days: int = 30, stock_name: str = ""):
    """生成HTML图表（TradingView风格）"""
    # 计算MACD（使用完整数据）
    macd_data = compute_macd(df['close'])
    df['dif'] = macd_data['dif']
    df['dea'] = macd_data['dea']
    df['macd'] = macd_data['macd']
    
    # 计算布林带（使用完整数据）
    boll_data = compute_bollinger(df['close'], period=20, std_dev=2.0)
    df['boll_mid'] = boll_data['boll_mid']
    df['boll_upper'] = boll_data['boll_upper']
    df['boll_lower'] = boll_data['boll_lower']
    
    # 计算成交量ZScore（使用完整数据，255日窗口）
    df['vol_zscore'] = compute_volume_zscore(df['volume'], win=255)
    
    # 只显示最后N天的数据
    df_display = df.tail(display_days).reset_index(drop=True)
    
    # 时间轴使用字符串避免间隙
    xcat = df_display['date'].dt.strftime('%Y-%m-%d').tolist()
    
    # 中国A股风格颜色（红涨绿跌）
    TV_BG = '#131722'
    TV_GRID = 'rgba(42, 46, 57, 0.5)'
    TV_TEXT = '#d1d4dc'
    TV_UP = '#ef5350'  # 红色上涨
    TV_DOWN = '#26a69a'  # 绿色下跌
    TV_VOLUME_UP = 'rgba(239, 83, 80, 0.5)'
    TV_VOLUME_DOWN = 'rgba(38, 166, 154, 0.5)'
    TV_DIF = '#2962ff'  # 蓝色
    TV_DEA = '#ff6d00'  # 橙色
    TV_BOLL = '#2962ff'  # 布林带颜色
    
    # 创建子图（5行：K线+布林带、主买主卖金额、主买主卖笔数、MACD、成交量ZScore）
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.42, 0.15, 0.15, 0.14, 0.14],
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
    )
    
    # 1. K线图
    fig.add_trace(
        go.Candlestick(
            x=xcat,
            open=df_display['open'],
            high=df_display['high'],
            low=df_display['low'],
            close=df_display['close'],
            increasing_line_color=TV_UP,
            decreasing_line_color=TV_DOWN,
            increasing_fillcolor=TV_UP,
            decreasing_fillcolor=TV_DOWN,
            showlegend=False,
            name='K线',
        ),
        row=1, col=1
    )
    
    # 添加布林带
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=df_display['boll_upper'],
            mode='lines',
            name='BOLL上轨',
            line=dict(color=TV_BOLL, width=1),
            showlegend=True,
            opacity=0.8,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=df_display['boll_mid'],
            mode='lines',
            name='BOLL中轨',
            line=dict(color='#ffd54f', width=1),
            showlegend=True,
            opacity=0.8,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=df_display['boll_lower'],
            mode='lines',
            name='BOLL下轨',
            line=dict(color=TV_BOLL, width=1),
            showlegend=True,
            opacity=0.8,
        ),
        row=1, col=1
    )
    
    # 2. 主买/主卖金额柱状图（并列放置，方便对比）
    fig.add_trace(
        go.Bar(
            x=xcat,
            y=df_display['buy_amount'] / 1e8,
            name='主买金额',
            marker_color=TV_UP,
            showlegend=True,
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=xcat,
            y=df_display['sell_amount'] / 1e8,
            name='主卖金额',
            marker_color=TV_DOWN,
            showlegend=True,
        ),
        row=2, col=1
    )
    
    # 添加5日平均线（颜色与柱体一致）
    buy_amount_ma5 = df_display['buy_amount'].rolling(window=5, min_periods=1).mean() / 1e8
    sell_amount_ma5 = df_display['sell_amount'].rolling(window=5, min_periods=1).mean() / 1e8
    
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=buy_amount_ma5,
            mode='lines',
            name='主买5日均',
            line=dict(color=TV_UP, width=1.5),
            showlegend=True,
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=sell_amount_ma5,
            mode='lines',
            name='主卖5日均',
            line=dict(color=TV_DOWN, width=1.5),
            showlegend=True,
        ),
        row=2, col=1
    )
    
    # 3. 主买/主卖笔数柱状图（并列放置，方便对比）
    fig.add_trace(
        go.Bar(
            x=xcat,
            y=df_display['buy_trades'],
            name='主买笔数',
            marker_color=TV_UP,  # 红色
            showlegend=True,
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(
            x=xcat,
            y=df_display['sell_trades'],
            name='主卖笔数',
            marker_color=TV_DOWN,  # 绿色
            showlegend=True,
        ),
        row=3, col=1
    )
    
    # 4. MACD
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=df_display['dif'],
            mode='lines',
            name='DIF',
            line=dict(color=TV_DIF, width=1.5),
            showlegend=True,
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=df_display['dea'],
            mode='lines',
            name='DEA',
            line=dict(color=TV_DEA, width=1.5),
            showlegend=True,
        ),
        row=4, col=1
    )
    # MACD柱状图
    macd_colors = [TV_UP if v >= 0 else TV_DOWN for v in df_display['macd']]
    fig.add_trace(
        go.Bar(
            x=xcat,
            y=df_display['macd'],
            name='MACD',
            marker_color=macd_colors,
            showlegend=True,
        ),
        row=4, col=1
    )
    
    # 5. 成交量ZScore（柱状图，淡蓝色）
    fig.add_trace(
        go.Bar(
            x=xcat,
            y=df_display['vol_zscore'],
            name='Vol ZScore',
            marker_color='#64b5f6',  # 淡蓝色
            showlegend=True,
        ),
        row=5, col=1
    )
    # 添加参考线 (0 / ±1 / ±2)
    for y0 in [0, 1, -1, 2, -2]:
        fig.add_hline(y=y0, row=5, col=1, line_width=1, opacity=0.35, line_color='rgba(255,255,255,0.3)')
    
    # 添加90%置信区间警戒线 (z = ±1.645)
    fig.add_hline(y=1.645, row=5, col=1, line_width=2, opacity=0.8, 
                  line_color='#ff9800', line_dash='dash')
    fig.add_hline(y=-1.645, row=5, col=1, line_width=2, opacity=0.8, 
                  line_color='#ff9800', line_dash='dash')
    
    # 标记超过警戒线的日期（成交量异常放大或缩小）
    zscore_90 = 1.645
    for idx, row_data in df_display.iterrows():
        z_val = row_data['vol_zscore']
        if pd.notna(z_val) and abs(z_val) > zscore_90:
            date_str = row_data['date'].strftime('%Y-%m-%d')
            color = TV_UP if z_val > 0 else TV_DOWN
            symbol_mark = '▲' if z_val > 0 else '▼'
            fig.add_annotation(
                x=date_str,
                y=z_val,
                text=symbol_mark,
                showarrow=False,
                font=dict(color=color, size=12),
                row=5, col=1,
            )
    
    # 添加信息面板
    last = df_display.iloc[-1]
    prev = df_display.iloc[-2] if len(df_display) > 1 else last
    change_pct = (last['close'] - prev['close']) / prev['close'] * 100
    change_symbol = '▲' if change_pct >= 0 else '▼'
    change_color = TV_UP if change_pct >= 0 else TV_DOWN
    
    # 显示名称：如果有股票名字则显示，否则只显示代码
    display_name = f"{stock_name} ({symbol})" if stock_name else symbol
    
    info_lines = [
        f"<b style='font-size:16px'>{display_name}</b>",
        f"<span style='color:{change_color};font-size:14px'>{last['close']:.2f} {change_symbol}{abs(change_pct):.2f}%</span>",
        f"",
        f"<b>主买/主卖金额（亿）</b>",
        f"主买: <span style='color:{TV_UP}'>{last['buy_amount']/1e8:.2f}</span>",
        f"主卖: <span style='color:{TV_DOWN}'>{last['sell_amount']/1e8:.2f}</span>",
        f"比值: {last['buy_sell_amount_ratio']:.4f}",
        f"",
        f"<b>主买/主卖笔数</b>",
        f"主买: <span style='color:{TV_UP}'>{last['buy_trades']:,}</span>",
        f"主卖: <span style='color:{TV_DOWN}'>{last['sell_trades']:,}</span>",
        f"比值: {last['buy_sell_trade_ratio']:.4f}",
        f"",
        f"<b>净买入</b>",
        f"<span style='color:{TV_UP if last['net_buy_amount'] >= 0 else TV_DOWN}'>{last['net_buy_amount']/1e8:.2f}亿</span>",
    ]
    
    fig.add_annotation(
        x=0.005, y=0.995,
        xref='paper', yref='paper',
        text='<br>'.join(info_lines),
        showarrow=False,
        align='left',
        bgcolor='rgba(19, 23, 34, 0.9)',
        bordercolor='rgba(42, 46, 57, 0.8)',
        borderwidth=1,
        font=dict(color=TV_TEXT, size=11, family='Trebuchet MS, Roboto, sans-serif'),
        row=1, col=1,
    )
    
    # 更新布局 - TradingView风格
    fig.update_layout(
        plot_bgcolor=TV_BG,
        paper_bgcolor=TV_BG,
        font=dict(color=TV_TEXT, family='Trebuchet MS, Roboto, sans-serif'),
        height=900,
        margin=dict(l=60, r=60, t=30, b=30),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='left',
            x=0.01,
            font=dict(size=10),
            bgcolor='rgba(19, 23, 34, 0)',
        ),
        hovermode='x unified',
        dragmode='zoom',
    )
    
    # 更新X轴
    fig.update_xaxes(
        showgrid=True,
        gridcolor=TV_GRID,
        gridwidth=0.5,
        type='category',
        categoryorder='array',
        categoryarray=xcat,
        showticklabels=True,
        tickangle=0,
        tickfont=dict(size=9),
        rangeslider_visible=False,
    )
    
    # 更新Y轴
    fig.update_yaxes(
        showgrid=True,
        gridcolor=TV_GRID,
        gridwidth=0.5,
        showticklabels=True,
        tickfont=dict(size=9),
        fixedrange=False,
    )
    
    # 设置Y轴标题
    fig.update_yaxes(title_text='', row=1, col=1)
    fig.update_yaxes(title_text='金额(亿)', row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text='笔数', row=3, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text='MACD', row=4, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text='Vol ZScore', row=5, col=1, title_font=dict(size=10))
    
    # K线图Y轴设置为非负范围（从最低价的95%开始）
    price_min = df_display['low'].min()
    price_max = df_display['high'].max()
    fig.update_yaxes(row=1, col=1, range=[price_min * 0.95, price_max * 1.05])
    
    # 只在最底部显示X轴标签
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)
    fig.update_xaxes(showticklabels=False, row=4, col=1)
    fig.update_xaxes(showticklabels=True, row=5, col=1)
    
    # 保存HTML
    fig.write_html(output_path, include_plotlyjs='cdn', config={
        'displayModeBar': True,
        'scrollZoom': True,
        'responsive': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    })
    print(f"\n✅ HTML已保存: {output_path}")
    
    # 保存图片
    if output_path.endswith('.html'):
        img_path = output_path[:-5] + '.png'
    else:
        img_path = output_path + '.png'
    
    try:
        fig.write_image(img_path, scale=2)
        print(f"✅ 图片已保存: {img_path}")
    except Exception as e:
        print(f"⚠️  图片保存失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="获取逐笔交易数据并分析主买主卖（30天汇总版）")
    parser.add_argument("--symbol", type=str, default="", help="股票代码")
    parser.add_argument("--days", type=int, default=30, help="获取最近N天数据（默认30天）")
    parser.add_argument("--output", type=str, default="", help="输出HTML文件路径")
    parser.add_argument("--batch", type=str, default="", help="批量处理：指定Excel文件路径")
    parser.add_argument("--output-dir", type=str, default="", help="批量处理输出目录")
    
    args = parser.parse_args()
    
    # 批量处理模式
    if args.batch:
        output_dir = args.output_dir or "/Users/zhenbao/Nextcloud/coding/交易/复盘"
        batch_process(args.batch, output_dir, args.days)
        return
    
    # 单只股票处理模式
    if not args.symbol:
        parser.print_help()
        print("\n错误: 请指定 --symbol 或 --batch 参数")
        return
    
    output_path = args.output
    if not output_path:
        output_path = f"{args.symbol}_tick_analysis.html"
    
    # 连接 pytdx
    api = connect_pytdx()
    
    try:
        # 获取股票名称
        stock_name = get_stock_name(api, args.symbol)
        display_name = f"{stock_name} ({args.symbol})" if stock_name else args.symbol
        print(f"\n股票: {display_name}")
        
        # 获取30天逐笔数据汇总
        df = get_30day_tick_summary(api, args.symbol, args.days)
        
        if df.empty:
            print("❌ 未获取到数据")
            return
        
        # 打印汇总信息
        print("\n" + "=" * 70)
        print(f"📊 {display_name} 最近 {args.days} 个交易日逐笔交易汇总")
        print("=" * 70)
        
        # 只统计最后N天的数据
        df_stats = df.tail(args.days)
        
        total_buy_amount = df_stats['buy_amount'].sum()
        total_sell_amount = df_stats['sell_amount'].sum()
        total_buy_trades = df_stats['buy_trades'].sum()
        total_sell_trades = df_stats['sell_trades'].sum()
        
        print(f"\n【汇总统计】（最近{args.days}个交易日）")
        print(f"  主买总金额: {total_buy_amount/1e8:.2f} 亿")
        print(f"  主卖总金额: {total_sell_amount/1e8:.2f} 亿")
        print(f"  主买/主卖金额比值: {total_buy_amount/total_sell_amount:.4f}")
        print(f"\n  主买总笔数: {total_buy_trades:,} 笔")
        print(f"  主卖总笔数: {total_sell_trades:,} 笔")
        print(f"  主买/主卖笔数比值: {total_buy_trades/total_sell_trades:.4f}")
        print(f"\n  净买入金额: {(total_buy_amount-total_sell_amount)/1e8:.2f} 亿")
        print("=" * 70)
        
        # 生成HTML
        build_html(df, args.symbol, output_path, display_days=args.days, stock_name=stock_name)
        
    finally:
        api.disconnect()
        print("\n✅ 已断开连接")


def get_stock_code_by_name(api: TdxHq_API, name: str, stock_cache_path: str = None) -> Optional[str]:
    """通过股票名称查找股票代码，优先从数据库缓存查找"""
    # 优先从数据库缓存查找
    try:
        from app.db import get_session
        from sqlalchemy import text
        
        with get_session() as session:
            sql = "SELECT ts_code FROM stock_concepts_cache WHERE name = :name"
            result = session.execute(text(sql), {"name": name})
            row = result.first()
            if row:
                ts_code = row[0]
                # ts_code 格式为 "000426.SZ"，提取代码部分
                return ts_code.split('.')[0]
    except Exception:
        pass  # 数据库查询失败，继续尝试从文件读取
    
    # 从本地缓存文件查找（向后兼容）
    if stock_cache_path and os.path.exists(stock_cache_path):
        cache_df = pd.read_excel(stock_cache_path)
        if 'name' in cache_df.columns and 'ts_code' in cache_df.columns:
            match = cache_df[cache_df['name'] == name]
            if not match.empty:
                ts_code = match.iloc[0]['ts_code']
                # ts_code 格式为 "000426.SZ"，提取代码部分
                code = ts_code.split('.')[0]
                return code
    
    # 从 pytdx 查找
    for market in [1, 0]:
        for start in range(0, 1000, 100):
            data = api.get_security_list(market, start)
            if not data:
                break
            for item in data:
                if item['name'] == name:
                    return item['code']
    return None


def load_stock_list(excel_path: str) -> List[Dict]:
    """从Excel文件加载股票列表，返回 [{'code': '000426', 'name': '兴业银锡'}, ...]"""
    df = pd.read_excel(excel_path)
    
    result = []
    
    # 检查是否有股票代码列
    if '股票代码' in df.columns:
        codes = df['股票代码'].tolist()
        names = df['股票名称'].tolist() if '股票名称' in df.columns else [''] * len(codes)
        for code, name in zip(codes, names):
            result.append({'code': str(code), 'name': str(name)})
    elif '股票名称' in df.columns:
        # 只有股票名称
        for name in df['股票名称'].tolist():
            result.append({'code': '', 'name': str(name)})
    else:
        # 尝试使用第一列
        for val in df.iloc[:, 0].tolist():
            val_str = str(val)
            if val_str.isdigit() or (len(val_str) == 6 and val_str[0].isdigit()):
                result.append({'code': val_str, 'name': ''})
            else:
                result.append({'code': '', 'name': val_str})
    
    return result


def batch_process(excel_path: str, output_dir: str, days: int = 30, stock_cache_path: str = None):
    """批量处理股票列表"""
    import os
    
    # 默认缓存文件路径
    if not stock_cache_path:
        stock_cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stock_concepts_cache.xlsx')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载股票列表
    stock_list = load_stock_list(excel_path)
    print(f"共加载 {len(stock_list)} 只股票")
    print("=" * 70)
    
    # 连接 pytdx
    api = connect_pytdx()
    
    try:
        success_count = 0
        fail_count = 0
        
        for idx, stock_info in enumerate(stock_list):
            stock_code = stock_info['code']
            stock_name_input = stock_info['name']
            
            print(f"\n[{idx+1}/{len(stock_list)}] 处理: {stock_name_input or stock_code}")
            print("-" * 50)
            
            # 确定股票代码和名称
            if stock_code:
                # 有股票代码，直接使用
                symbol = stock_code
                stock_name = stock_name_input or get_stock_name(api, symbol)
            else:
                # 只有股票名称，尝试查找代码
                symbol = get_stock_code_by_name(api, stock_name_input, stock_cache_path)
                if not symbol:
                    print(f"❌ 未找到股票: {stock_name_input}")
                    fail_count += 1
                    continue
                stock_name = stock_name_input
            
            display_name = f"{stock_name} ({symbol})" if stock_name else symbol
            print(f"股票: {display_name}")
            
            # 获取数据
            df = get_30day_tick_summary(api, symbol, days)
            
            if df.empty:
                print(f"❌ 未获取到数据")
                fail_count += 1
                continue
            
            # 生成HTML（使用股票名称命名）
            safe_name = stock_name or symbol
            # 移除文件名中的非法字符
            safe_name = safe_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_path = os.path.join(output_dir, f"{safe_name}_tick_analysis.html")
            build_html(df, symbol, output_path, display_days=days, stock_name=stock_name)
            
            success_count += 1
        
        print("\n" + "=" * 70)
        print(f"批量处理完成")
        print(f"  成功: {success_count} 只")
        print(f"  失败: {fail_count} 只")
        print(f"  输出目录: {output_dir}")
        print("=" * 70)
        
    finally:
        api.disconnect()
        print("\n✅ 已断开连接")


if __name__ == "__main__":
    main()
