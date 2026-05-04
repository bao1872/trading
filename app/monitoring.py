# -*- coding: utf-8 -*-
"""
自选股监控与飞书通知

Purpose: 扫描 stock_watchlist 表中 is_monitored=TRUE 的自选股，检测5m/15m/60m级别的MACD/Hist顶底背离、DSA VWAP触及、PAVP价格穿越，通过飞书推送结果
Inputs:
    - stock_watchlist 表（is_monitored=TRUE 的自选股）
    - pytdx 行情数据（5分钟/15分钟/60分钟）
Outputs:
    - 飞书消息推送（包含股票信息、背离检测、PAVP穿越和DSA VWAP检测结果）
How to Run:
    python app/monitoring.py --freq 5m
    python app/monitoring.py --freq 15m
    python app/monitoring.py --freq 60m
Examples:
    python app/monitoring.py --freq 60m
Side Effects:
    - 读取 stock_watchlist 表数据（is_monitored=TRUE）
    - 通过 pytdx 获取行情数据
    - 发送飞书消息通知
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text

from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_USER_ID
from datasource.database import get_session
from datasource.pytdx_client import connect_pytdx, PERIOD_MAP
from features.divergence_many_plotly import DivConfig, run_divergence_engine
from features.merged_dsa_atr_rope_bb_factors import compute_dsa, DSAConfig

# 全局 pytdx 连接，只初始化一次
_API = None


def get_api():
    """获取全局 pytdx 连接，单例模式"""
    global _API
    if _API is None:
        _API = connect_pytdx()
    return _API


def close_api():
    """关闭 pytdx 连接"""
    global _API
    if _API is not None:
        _API.disconnect()
        _API = None


def fetch_all_kline(ts_codes: List[str], freq: str, bars: int = 500, max_time: datetime = None) -> Dict[str, pd.DataFrame]:
    """批量获取多只股票的K线数据，使用全局 pytdx 连接
    Args:
        ts_codes: 股票代码列表
        freq: 周期（5m/15m/60m）
        bars: 回溯bar数量
        max_time: 数据截止时间，只获取该时间之前的数据（不含）
    Returns:
        {ts_code: DataFrame} 映射
    """
    api = get_api()
    cat = PERIOD_MAP.get(freq.lower())
    if cat is None:
        raise ValueError(f"不支持的 freq: {freq}")

    result = {}
    for ts_code in ts_codes:
        try:
            mkt = 1 if ts_code.startswith("6") else 0
            all_bars = []
            page = 0
            page_size = 800
            max_pages = (bars + page_size - 1) // page_size + 1

            while page < max_pages:
                recs = api.get_security_bars(cat, mkt, ts_code, page * page_size, page_size)
                if not recs:
                    break
                df = pd.DataFrame(recs)
                if df.empty:
                    break

                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
                    df = df.set_index("datetime")

                if "vol" in df.columns:
                    df = df.rename(columns={"vol": "volume"})

                all_bars.append(df)
                if len(recs) < page_size:
                    break
                page += 1

            if all_bars:
                kline = pd.concat(all_bars).sort_index()
                if max_time is not None:
                    kline = kline[kline.index < max_time]
                kline = kline.tail(bars)
                for c in ["open", "high", "low", "close", "volume"]:
                    kline[c] = pd.to_numeric(kline[c], errors="coerce")
                kline = kline.dropna(subset=["open", "high", "low", "close", "volume"])
                if not kline.empty:
                    result[ts_code] = kline
        except Exception:
            pass

    return result


def detect_divergences(df: pd.DataFrame) -> List[str]:
    """检测MACD和Hist的顶底背离（常规/隐藏），只检测最后一个bar是否确认了背离
    Args:
        df: K线数据
    Returns:
        list of divergence descriptions (e.g., ["MACD常规底背离", "Hist隐藏顶背离"])
    """
    if len(df) < 50:
        return []

    cfg = DivConfig()
    cfg.calcmacd = True
    cfg.calcmacda = True
    cfg.calcrsi = False
    cfg.calcobv = False
    cfg.calcvwmacd = False
    cfg.calccmf = False
    cfg.calcmfi = False
    cfg.calcext = False
    cfg.calcstoc = False
    cfg.calccci = False
    cfg.calcmom = False
    cfg.showindis = "Full"
    cfg.searchdiv = "Regular/Hidden"
    cfg.only_pos_divs = False
    cfg.showlast = True

    # 背离类型颜色映射
    # pos_reg_div_col: rgba(245, 222, 66, 1.0) - 黄色 - 常规底背离
    # pos_hid_div_col: rgba(0, 214, 106, 1.0) - 绿色 - 隐藏底背离
    # neg_reg_div_col: rgba(54, 40, 160, 1.0) - 深蓝 - 常规顶背离
    # neg_hid_div_col: rgba(255, 87, 87, 1.0) - 红色 - 隐藏顶背离
    POS_REG_COL = "rgba(245, 222, 66"
    POS_HID_COL = "rgba(0, 214, 106"
    NEG_REG_COL = "rgba(54, 40, 160"
    NEG_HID_COL = "rgba(255, 87, 87"

    try:
        pos_lines, neg_lines, pos_labels, neg_labels, _, _ = run_divergence_engine(df, cfg)

        # 最后一个bar的时间
        last_bar_time = df.index[-1]
        last_bar_str = pd.Timestamp(last_bar_time).strftime("%Y-%m-%d %H:%M")

        results = []

        for label in pos_labels:
            ts = label[0]
            text_content = label[2]
            label_color = label[3] if len(label) > 3 else ""
            if ts == last_bar_str:
                # 根据颜色判断是常规还是隐藏背离
                if POS_HID_COL in label_color:
                    div_type = "隐藏"
                else:
                    div_type = "常规"

                if "MACD" in text_content:
                    results.append(f"MACD{div_type}底背离")
                elif "Hist" in text_content:
                    results.append(f"Hist{div_type}底背离")

        for label in neg_labels:
            ts = label[0]
            text_content = label[2]
            label_color = label[3] if len(label) > 3 else ""
            if ts == last_bar_str:
                # 根据颜色判断是常规还是隐藏背离
                if NEG_HID_COL in label_color:
                    div_type = "隐藏"
                else:
                    div_type = "常规"

                if "MACD" in text_content:
                    results.append(f"MACD{div_type}顶背离")
                elif "Hist" in text_content:
                    results.append(f"Hist{div_type}顶背离")

        return list(set(results))
    except Exception:
        return []


def detect_dsa_vwap(df: pd.DataFrame) -> str:
    """检测最后一个bar是否高低点跨越VWAP线
    Args:
        df: K线数据（需包含open, high, low, close, volume）
    Returns:
        str: 触发原因描述，如 'DSA VWAP触及 (dir=1)'；无触发返回空字符串
    """
    if len(df) < 50:
        return ""

    try:
        dsa_df, _, _ = compute_dsa(df, DSAConfig())
        if dsa_df.empty:
            return ""

        latest = dsa_df.iloc[-1]
        raw_dir = latest.get('DSA_DIR')
        raw_vwap = latest.get('DSA_VWAP')

        if pd.isna(raw_dir) or pd.isna(raw_vwap):
            return ""

        dir_val = int(raw_dir)
        vwap_val = float(raw_vwap)

        last_bar = df.iloc[-1]
        high = last_bar['high']
        low = last_bar['low']

        if low <= vwap_val <= high:
            return f"DSA VWAP触及 (dir={dir_val})"

    except Exception as e:
        print(f"DSA VWAP检测异常: {e}")

    return ""


def get_monitored_stocks(session) -> list:
    """获取被勾选监控的股票列表（is_monitored=TRUE）"""
    sql = text("""
        SELECT ts_code, stock_name, bsm_event, bsm_event_date,
               pavp_prices, dsa_dir, dsa_vwap
        FROM stock_watchlist
        WHERE is_monitored = TRUE
        ORDER BY sort_order, added_date DESC
    """)
    rows = session.execute(sql).mappings().all()
    stocks = []
    for row in rows:
        stock = dict(row)
        stock['ts_code'] = str(stock['ts_code']).zfill(6)
        # 解析 pavp_prices 字符串
        pavp_str = stock.get('pavp_prices', '')
        stock['pavp'] = parse_pavp_prices(pavp_str)
        stocks.append(stock)
    return stocks


def parse_pavp_prices(pavp_str: str) -> dict:
    """解析PAVP价格字符串，如 'VAH:43.37,POC:24.47,VAL:18.91'"""
    result = {}
    if not pavp_str:
        return result
    for part in pavp_str.split(','):
        if ':' in part:
            key, val = part.split(':', 1)
            try:
                result[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return result


def detect_price_crossings(df: pd.DataFrame, stock: dict) -> list:
    """检测价格上穿/下穿 PAVP(VAH/VAL) 和 DSA_VWAP

    检测逻辑：
    - 上穿：前一根K线收盘价 < 价格线 且 当前K线收盘价 >= 价格线
    - 下穿：前一根K线收盘价 > 价格线 且 当前K线收盘价 <= 价格线
    """
    if len(df) < 2:
        return []

    triggers = []
    prev_close = df['close'].iloc[-2]
    curr_close = df['close'].iloc[-1]

    pavp = stock.get('pavp', {})
    vah = pavp.get('VAH')
    val = pavp.get('VAL')
    dsa_vwap = stock.get('dsa_vwap')

    # VAH
    if vah is not None and pd.notna(vah):
        if prev_close < vah <= curr_close:
            triggers.append(f"上穿VAH({vah:.2f})")
        elif prev_close > vah >= curr_close:
            triggers.append(f"下穿VAH({vah:.2f})")

    # VAL
    if val is not None and pd.notna(val):
        if prev_close < val <= curr_close:
            triggers.append(f"上穿VAL({val:.2f})")
        elif prev_close > val >= curr_close:
            triggers.append(f"下穿VAL({val:.2f})")

    # DSA_VWAP
    if dsa_vwap is not None and pd.notna(dsa_vwap):
        if prev_close < dsa_vwap <= curr_close:
            triggers.append(f"上穿DSA_VWAP({dsa_vwap:.2f})")
        elif prev_close > dsa_vwap >= curr_close:
            triggers.append(f"下穿DSA_VWAP({dsa_vwap:.2f})")

    return triggers


import argparse


VALID_FREQS = ["5m", "15m", "60m"]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="精选股票池背离检测")
    parser.add_argument(
        "--freq",
        type=str,
        choices=VALID_FREQS,
        required=True,
        help="检测周期（5m/15m/60m）",
    )
    return parser.parse_args()


FREQ_EMOJI = {"5m": "⚡", "15m": "📈", "60m": "📊"}


def get_trigger_emoji(trigger: str) -> str:
    """根据触发类型返回对应的 emoji"""
    if "底背离" in trigger:
        return "🟢"
    elif "顶背离" in trigger:
        return "🔴"
    elif "VWAP触及" in trigger:
        if "dir=1" in trigger:
            return "🔵"
        elif "dir=-1" in trigger:
            return "🟠"
    return "📌"


def generate_watchlist_monitor_report(freq: str, triggered_stocks: List[Dict]) -> str:
    """生成自选股监控报告（包含日线价格穿越、本周期VWAP触及、顶底背离信号）
    Args:
        freq: 周期(5m/15m/60m)
        triggered_stocks: 触发信号的股票列表
    Returns:
        Markdown 格式的报告
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    freq_emoji = FREQ_EMOJI.get(freq, "📊")

    lines = [
        f"{freq_emoji} **自选股监控报告 [{freq}]** ({now_str})",
        "",
        "**📈 概览统计**",
    ]

    # 统计各类信号
    total_count = len(triggered_stocks)
    # 日线价格穿越（来自数据库存储的日线PAVP/DSA_VWAP）
    vah_count = sum(1 for item in triggered_stocks for t in item['triggers'] if 'VAH' in t)
    val_count = sum(1 for item in triggered_stocks for t in item['triggers'] if 'VAL' in t)
    dsa_vwap_cross_count = sum(1 for item in triggered_stocks for t in item['triggers'] if 'DSA_VWAP' in t and ('上穿' in t or '下穿' in t))
    # 本周期VWAP触及（来自实时计算的DSA指标）
    vwap_touch_count = sum(1 for item in triggered_stocks for t in item['triggers'] if 'VWAP触及' in t)
    # 背离信号
    div_count = sum(1 for item in triggered_stocks for t in item['triggers'] if '背离' in t)

    lines.extend([
        f"- **触发总数:** {total_count} 只",
        f"- **日线VAH穿越:** {vah_count} 次",
        f"- **日线VAL穿越:** {val_count} 次",
        f"- **日线DSA_VWAP穿越:** {dsa_vwap_cross_count} 次",
        f"- **本周期VWAP触及:** {vwap_touch_count} 次",
        f"- **背离信号:** {div_count} 个",
        "",
        "**🔥 触发详情**",
        "",
    ])

    for idx, item in enumerate(triggered_stocks, 1):
        stock = item['stock']
        triggers = item['triggers']

        stock_name = stock.get('stock_name', '-')
        ts_code = stock.get('ts_code', '-')

        lines.append(f"**{idx}. {stock_name} ({ts_code})**")

        # 分类显示信号
        # 日线价格穿越（VAH/VAL/DSA_VWAP）
        daily_cross_signals = [t for t in triggers if ('VAH' in t or 'VAL' in t or 'DSA_VWAP' in t) and ('上穿' in t or '下穿' in t)]
        # 本周期VWAP触及
        vwap_touch_signals = [t for t in triggers if 'VWAP触及' in t]
        # 背离信号
        div_signals = [t for t in triggers if '背离' in t]

        # 日线价格穿越
        if daily_cross_signals:
            cross_lines = []
            for t in daily_cross_signals:
                if '上穿' in t:
                    cross_lines.append(f"🟢 {t}")
                elif '下穿' in t:
                    cross_lines.append(f"🔴 {t}")
            lines.append(f"- **日线价格穿越:** {' | '.join(cross_lines)}")

        # 本周期VWAP触及
        if vwap_touch_signals:
            touch_lines = []
            for t in vwap_touch_signals:
                if 'dir=1' in t:
                    touch_lines.append(f"🔵 {t} (多头)")
                elif 'dir=-1' in t:
                    touch_lines.append(f"🟠 {t} (空头)")
                else:
                    touch_lines.append(f"📊 {t}")
            lines.append(f"- **本周期VWAP触及:** {' | '.join(touch_lines)}")

        # 背离信号
        if div_signals:
            div_lines = []
            for t in div_signals:
                if '底背离' in t:
                    div_lines.append(f"🟢 {t}")
                elif '顶背离' in t:
                    div_lines.append(f"🔴 {t}")
                else:
                    div_lines.append(f"📊 {t}")
            lines.append(f"- **背离信号:** {' | '.join(div_lines)}")

        lines.append("")

    return "\n".join(lines)


def send_feishu_message(content: str):
    """发送飞书卡片消息（Markdown 格式）"""
    try:
        import json
        import requests

        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        token_resp = requests.post(token_url, json={
            "app_id": FEISHU_APP_ID,
            "app_secret": FEISHU_APP_SECRET,
        })
        token = token_resp.json().get("tenant_access_token")
        if not token:
            print(f"飞书获取token失败: {token_resp.text}")
            return

        msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {"Authorization": f"Bearer {token}"}

        # 使用 interactive 消息类型（卡片消息）
        card_content = {
            "config": {
                "wide_screen_mode": True
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": content
                }
            ]
        }

        body = json.dumps({
            "receive_id": FEISHU_USER_ID,
            "msg_type": "interactive",
            "content": json.dumps(card_content),
        })
        msg_resp = requests.post(
            msg_url,
            headers=headers,
            params={"receive_id_type": "user_id"},
            data=body,
        )
        if msg_resp.status_code != 200:
            print(f"飞书消息发送失败: {msg_resp.text}")
        else:
            result = msg_resp.json()
            if result.get("code") == 0:
                print("飞书卡片消息发送成功")
            else:
                print(f"飞书消息发送失败: {result.get('msg')}")
    except Exception as e:
        print(f"飞书通知异常: {e}")


def run_watchlist_monitor(freq: str):
    """执行自选股监控任务（只监控is_monitored=TRUE的股票）
    Args:
        freq: 检测周期（5m/15m/60m）
    """
    if freq not in VALID_FREQS:
        raise ValueError(f"不支持的 freq: {freq}，可选: {VALID_FREQS}")

    print(f"[{datetime.now()}] 开始自选股监控 (周期: {freq})")

    with get_session() as session:
        stocks = get_monitored_stocks(session)  # 只获取被勾选的股票

    if not stocks:
        print("无被勾选监控的股票")
        return

    print(f"被勾选监控的股票数量: {len(stocks)}")

    ts_codes = [s["ts_code"] for s in stocks]
    now = datetime.now().replace(second=0, microsecond=0)
    # 根据周期设置缓冲时间，确保获取完整的最新K线数据
    # 5m周期缓冲5分钟，15m周期缓冲15分钟，60m周期缓冲60分钟
    freq_minutes = {"5m": 5, "15m": 15, "60m": 60}
    buffer_minutes = freq_minutes.get(freq, 5)
    max_time = now + pd.Timedelta(minutes=buffer_minutes)

    print(f"开始获取 {freq} 行情数据...")
    kline_data = fetch_all_kline(ts_codes, freq, bars=500, max_time=max_time)
    print(f"{freq} 获取完成: {len(kline_data)}/{len(ts_codes)} 只股票")

    print("开始检测信号...")

    triggered_stocks = []

    for stock in stocks:
        ts_code = stock["ts_code"]
        df = kline_data.get(ts_code)
        if df is None or df.empty:
            continue

        triggers = []

        # 1. 价格穿越检测（新增）
        crossings = detect_price_crossings(df, stock)
        if crossings:
            triggers.extend(crossings)

        # 2. 背离检测（原有保留）
        divs = detect_divergences(df)
        if divs:
            for d in divs:
                triggers.append(f"{freq} {d}")

        # 3. DSA VWAP触及检测（原有保留）
        dsa_trigger = detect_dsa_vwap(df)
        if dsa_trigger:
            triggers.append(dsa_trigger)

        if triggers:
            triggered_stocks.append({'stock': stock, 'triggers': triggers})

    # 发送报告
    if triggered_stocks:
        print(f"检测到 {len(triggered_stocks)} 只股票有触发信号")
        report = generate_watchlist_monitor_report(freq, triggered_stocks)
        print("发送监控报告...")
        send_feishu_message(report)
    else:
        print("无触发信号")

    print(f"[{datetime.now()}] 检测完成")
    close_api()


def main():
    args = parse_args()
    run_watchlist_monitor(args.freq)


if __name__ == "__main__":
    main()
