# -*- coding: utf-8 -*-
"""
自选股监控与飞书通知

Purpose: 从 stock_watchlist 获取自选股列表，检测5m/15m/60m级别的MACD/Hist顶底背离、
         DSA VWAP触及，通过飞书推送结果（含优先级/核心驱动/4模型评分/涨跌幅等）
Inputs:
    - stock_watchlist 表（自选股列表，关联 stock_pools 和 stop_loss_predictions）
    - pytdx 行情数据（5分钟/15分钟/60分钟/日线）
Outputs:
    - 飞书消息推送（包含股票信息、背离检测和DSA VWAP检测结果）
How to Run:
    python app/monitoring.py --freq 5m
    python app/monitoring.py --freq 15m
    python app/monitoring.py --freq 60m
Examples:
    python app/monitoring.py --freq 60m
Side Effects:
    - 读取 stock_watchlist / stock_pools / stop_loss_predictions 表
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
    """获取全局 pytdx 连接，单例模式，自动检测断线重连"""
    global _API
    if _API is not None:
        try:
            _API.get_security_count(1)
        except Exception:
            print("pytdx 连接已断开，尝试重连...")
            try:
                _API.disconnect()
            except Exception:
                pass
            _API = None

    if _API is None:
        _API = connect_pytdx()
    return _API


def close_api():
    """关闭 pytdx 连接"""
    global _API
    if _API is not None:
        try:
            _API.disconnect()
        except Exception:
            pass
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
        except Exception as e:
            print(f"获取 {ts_code} {freq} K线失败: {e}")

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
    """获取自选股列表（从 stock_watchlist，关联 stock_pools 和 stop_loss_predictions）

    Returns:
        [{'ts_code': ..., 'stock_name': ..., 'priority': ..., 'weighted_score': ...,
          'core_driver': ..., 'second_deriv_type': ..., 'key_strength': ..., 'main_risk': ...,
          'total_market_cap': ..., 'pred_sell_reg': ..., 'pred_sell_cls': ...,
          'pred_buy_reg': ..., 'pred_buy_cls': ...}, ...]
    """
    sql = text("""
        SELECT
            w.ts_code,
            w.stock_name,
            w.priority,
            w.weighted_score,
            w.core_driver,
            w.second_deriv_type,
            w.key_strength,
            w.main_risk,
            p.total_market_cap / 100000000.0 AS total_market_cap,
            pred.pred_sell_reg,
            pred.pred_sell_cls,
            pred.pred_buy_reg,
            pred.pred_buy_cls
        FROM stock_watchlist w
        LEFT JOIN stock_pools p ON p.ts_code = w.ts_code
        LEFT JOIN LATERAL (
            SELECT pred_sell_reg, pred_sell_cls, pred_buy_reg, pred_buy_cls
            FROM stop_loss_predictions
            WHERE ts_code = w.ts_code
              AND profile IN ('production', 'position')
              AND prediction_date = (
                  SELECT MAX(prediction_date) FROM stop_loss_predictions
                  WHERE profile IN ('production', 'position')
              )
            ORDER BY obs_date DESC
            LIMIT 1
        ) pred ON TRUE
        ORDER BY w.sort_order, w.weighted_score DESC
    """)
    result = session.execute(sql).mappings().all()
    stocks = []
    for row in result:
        stocks.append({
            'ts_code': str(row['ts_code']).split('.')[0].zfill(6),
            'ts_code_full': str(row['ts_code']),
            'stock_name': row['stock_name'] or '',
            'priority': row['priority'] or '',
            'weighted_score': float(row['weighted_score']) if row['weighted_score'] else 0.0,
            'core_driver': row['core_driver'] or '',
            'second_deriv_type': row['second_deriv_type'] or '',
            'key_strength': row['key_strength'] or '',
            'main_risk': row['main_risk'] or '',
            'total_market_cap': round(float(row['total_market_cap']), 1) if row['total_market_cap'] else 0.0,
            'pred_sell_reg': round(float(row['pred_sell_reg']), 3) if row['pred_sell_reg'] is not None else None,
            'pred_sell_cls': round(float(row['pred_sell_cls']), 3) if row['pred_sell_cls'] is not None else None,
            'pred_buy_reg': round(float(row['pred_buy_reg']), 3) if row['pred_buy_reg'] is not None else None,
            'pred_buy_cls': round(float(row['pred_buy_cls']), 3) if row['pred_buy_cls'] is not None else None,
        })
    return stocks


def compute_daily_change_pct(ts_codes: List[str]) -> Dict[str, float]:
    """通过 pytdx 日K线计算当日涨跌幅

    Args:
        ts_codes: 纯6位股票代码列表（如 ['300394', '300502']）
    Returns:
        {ts_code: change_pct} 映射，涨跌幅百分比
    """
    kline_data = fetch_all_kline(ts_codes, 'd', bars=5)
    result = {}
    for code, df in kline_data.items():
        if df is None or len(df) < 2:
            continue
        try:
            prev_close = float(df.iloc[-2]['close'])
            cur_close = float(df.iloc[-1]['close'])
            if prev_close > 0:
                result[code] = round((cur_close - prev_close) / prev_close * 100, 2)
        except (IndexError, ValueError, ZeroDivisionError):
            pass
    return result


import argparse


VALID_FREQS = ["5m", "15m", "60m"]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="自选股背离检测（stock_watchlist）")
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


def generate_monitor_report(freq: str, all_stocks: List[Dict], triggered_stocks: List[Dict]) -> str:
    """生成自选股监控报告

    Args:
        freq: 周期(5m/15m/60m)
        all_stocks: 全部自选股列表（含 change_pct）
        triggered_stocks: 触发信号的股票列表
    Returns:
        Markdown 格式的报告
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    freq_emoji = FREQ_EMOJI.get(freq, "📊")

    triggered_codes = set()
    vwap_touch_count = 0
    div_count = 0
    for item in triggered_stocks:
        code = item['stock']['ts_code']
        triggered_codes.add(code)
        for t in item['triggers']:
            if 'VWAP触及' in t:
                vwap_touch_count += 1
            if '背离' in t:
                div_count += 1

    priority_counts = {}
    for s in all_stocks:
        p = s.get('priority', '')
        if p:
            priority_counts[p] = priority_counts.get(p, 0) + 1

    priority_summary = ' | '.join(f"{p}: {c}" for p, c in sorted(priority_counts.items(), key=lambda x: {'S': 0, 'S-': 1, 'A+': 2, 'A': 3, 'A-': 4, 'B+': 5, 'B': 6}.get(x[0], 99)))

    lines = [
        f"{freq_emoji} **自选股监控报告 [{freq}]** ({now_str})",
        "",
        f"**📊 概览**",
        f"- 自选股: {len(all_stocks)} 只 | {priority_summary}",
        f"- 触发信号: {len(triggered_stocks)} 只 | VWAP触及: {vwap_touch_count} | 背离: {div_count}",
        "",
    ]

    if triggered_stocks:
        lines.append("**⚠️ 触发信号**")
        lines.append("")
        for item in triggered_stocks:
            stock = item['stock']
            triggers = item['triggers']
            name = stock.get('stock_name', '-')
            code = stock.get('ts_code', '-')
            priority = stock.get('priority', '')
            score = stock.get('weighted_score', 0)
            driver = stock.get('core_driver', '')
            cap = stock.get('total_market_cap', 0)
            change = stock.get('change_pct', 0)
            deriv = stock.get('second_deriv_type', '')
            strength = stock.get('key_strength', '')
            risk = stock.get('main_risk', '')

            change_str = f"+{change}" if change > 0 else str(change)
            lines.append(f"**{name} ({code})**  `{priority}`  `{score}分`")
            lines.append(f"驱动: {driver} | 市值: {cap:.0f}亿 | 涨跌: {change_str}%")

            sr = stock.get('pred_sell_reg')
            sc = stock.get('pred_sell_cls')
            br = stock.get('pred_buy_reg')
            bc = stock.get('pred_buy_cls')
            model_parts = []
            if sr is not None:
                model_parts.append(f"Sell回归 {sr:.3f}")
            if sc is not None:
                model_parts.append(f"Sell分类 {sc:.3f}")
            if br is not None:
                model_parts.append(f"Buy回归 {br:.3f}")
            if bc is not None:
                model_parts.append(f"Buy分类 {bc:.3f}")
            if model_parts:
                lines.append(f"4模型: {' | '.join(model_parts)}")

            if deriv:
                lines.append(f"二阶导: {deriv}")
            if strength:
                lines.append(f"强点: {strength}")
            if risk:
                lines.append(f"风险: {risk}")

            trigger_parts = []
            for t in triggers:
                if '底背离' in t:
                    trigger_parts.append(f"🟢 {t}")
                elif '顶背离' in t:
                    trigger_parts.append(f"🔴 {t}")
                elif 'VWAP触及' in t:
                    if 'dir=1' in t:
                        trigger_parts.append(f"🔵 {t}")
                    elif 'dir=-1' in t:
                        trigger_parts.append(f"🟠 {t}")
                    else:
                        trigger_parts.append(f"📌 {t}")
                else:
                    trigger_parts.append(f"📌 {t}")
            lines.append(f"信号: {' | '.join(trigger_parts)}")
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


def run_monitor(freq: str):
    """执行自选股监控任务（从 stock_watchlist 获取全部自选股）

    Args:
        freq: 检测周期（5m/15m/60m）
    """
    if freq not in VALID_FREQS:
        raise ValueError(f"不支持的 freq: {freq}，可选: {VALID_FREQS}")

    print(f"[{datetime.now()}] 开始自选股监控 (周期: {freq})")

    with get_session() as session:
        stocks = get_monitored_stocks(session)

    if not stocks:
        print("无自选股")
        return

    print(f"自选股数量: {len(stocks)}")

    ts_codes = [s["ts_code"] for s in stocks]

    print("获取日K线计算涨跌幅...")
    change_map = compute_daily_change_pct(ts_codes)
    for s in stocks:
        s['change_pct'] = change_map.get(s['ts_code'], 0.0)

    now = datetime.now().replace(second=0, microsecond=0)
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

        divs = detect_divergences(df)
        if divs:
            for d in divs:
                triggers.append(f"{freq} {d}")

        dsa_trigger = detect_dsa_vwap(df)
        if dsa_trigger:
            triggers.append(dsa_trigger)

        if triggers:
            triggered_stocks.append({'stock': stock, 'triggers': triggers})

    print(f"检测到 {len(triggered_stocks)} 只股票有触发信号")

    if triggered_stocks:
        report = generate_monitor_report(freq, stocks, triggered_stocks)
        print("发送监控报告...")
        send_feishu_message(report)
    else:
        print("无触发信号，不推送")

    print(f"[{datetime.now()}] 检测完成")
    close_api()


def main():
    args = parse_args()
    run_monitor(args.freq)


if __name__ == "__main__":
    main()
