# -*- coding: utf-8 -*-
"""
布林带 + 节点集群 自选股监控与飞书通知

Purpose: 从 stock_watchlist 获取自选股列表，检测日线布林带触及（上轨/中轨/下轨）
         和日线节点集群触及（peak_levels），触发时发送飞书卡片通知+行情图片。
         信号检测基于上一根已完成bar，排除当前未完成bar避免信号闪烁。
Inputs:
    - stock_watchlist 表（自选股列表，关联 stock_pools 和 stop_loss_predictions）
    - pytdx 行情数据（日线 + 15分钟线 + 1分钟线）
Outputs:
    - 飞书卡片消息（颜色编码 + 信号详情 + 布林带快照）
    - 飞书图片消息（K线+布林带+节点集群 PNG图表）
How to Run:
    python app/monitoring.py                  # 单次执行监控
    python app/monitoring.py --dry-run        # 干跑模式，不发送通知
    python app/monitoring.py --test           # 测试模式，检测最后一次触发并推送图片
    python app/monitoring.py --schedule        # 启动持续监控调度器（交易时段内循环执行）
Examples:
    python app/monitoring.py
    python app/monitoring.py --dry-run
    python app/monitoring.py --test
    python app/monitoring.py --schedule
Side Effects:
    - 读取 stock_watchlist / stock_pools / stop_loss_predictions 表
    - 通过 pytdx 获取行情数据
    - 发送飞书消息通知（卡片+图片）
"""
import os
import sys
import tempfile
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "monitoring.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("monitoring")
from sqlalchemy import text

from app.feishu_notifier import FeishuNotifier
from datasource.database import get_session
from datasource.pytdx_client import connect_pytdx, PERIOD_MAP
from features.bollinger_features_plotly import bollinger
from features.luxalgo_volume_profile_pytdx_15m_aligned import (
    compute_volume_profile,
    VolumeProfileConfig,
    VolumeProfileResult,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 布林带参数
BB_WIN = 20
BB_K = 2.0

# 触发类型
BB_UPPER_TOUCH = "bb_upper_touch"
BB_MID_TOUCH = "bb_mid_touch"
BB_LOWER_TOUCH = "bb_lower_touch"
NODE_CLUSTER_TOUCH = "node_cluster_touch"

# 触发类型中文映射
TRIGGER_LABEL = {
    BB_UPPER_TOUCH: "布林上轨触及",
    BB_MID_TOUCH: "布林中轨触及",
    BB_LOWER_TOUCH: "布林下轨触及",
    NODE_CLUSTER_TOUCH: "节点集群触及",
}

# 事件颜色映射（用于图表标记和消息语义）
EVENT_COLOR_MAP = {
    BB_UPPER_TOUCH:     {"color": "#F44336", "label": "布林上轨触及", "emoji": "🔴", "severity": "danger"},
    BB_MID_TOUCH:       {"color": "#FF9800", "label": "布林中轨触及", "emoji": "🟠", "severity": "warn"},
    BB_LOWER_TOUCH:     {"color": "#4CAF50", "label": "布林下轨触及", "emoji": "🟢", "severity": "info"},
    NODE_CLUSTER_TOUCH: {"color": "#9C27B0", "label": "节点集群触及", "emoji": "🟣", "severity": "warn"},
}

# 飞书卡片header颜色模板
SEVERITY_TEMPLATE = {"danger": "red", "warn": "orange", "info": "green"}

# 触发类型对应 emoji
TRIGGER_EMOJI = {k: v["emoji"] for k, v in EVENT_COLOR_MAP.items()}

# Volume Profile 默认参数
VP_LOOKBACK = 360
VP_ROWS = 100
VP_VALUE_AREA_PCT = 0.70
VP_PEAK_DETECTION_PCT = 0.09
VP_NODE_THRESHOLD_PCT = 0.01

# 通知冷却：同一信号在冷却期内不重复推送
NOTIFY_COOLDOWN_SECONDS = 600  # 10分钟
_last_notified: Dict[str, float] = {}


def _should_notify(ts_code: str, trigger_type: str, boundary: float) -> bool:
    """判断是否应该发送通知（冷却期内不重复）"""
    import time as _time
    key = f"{ts_code}:{trigger_type}:{boundary:.2f}"
    now = _time.time()
    last = _last_notified.get(key, 0)
    if now - last < NOTIFY_COOLDOWN_SECONDS:
        return False
    _last_notified[key] = now
    return True

# 全局 pytdx 连接，只初始化一次
_API = None

# 飞书通知器
_notifier = FeishuNotifier()


# ---------------------------------------------------------------------------
# pytdx 连接管理
# ---------------------------------------------------------------------------

def get_api():
    """获取全局 pytdx 连接，单例模式，自动检测断线重连"""
    global _API
    if _API is not None:
        try:
            _API.get_security_count(1)
        except Exception:
            logger.warning("pytdx 连接已断开，尝试重连...")
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


# ---------------------------------------------------------------------------
# K线数据获取
# ---------------------------------------------------------------------------

def _get_completed_bar_index(df: pd.DataFrame, freq: str) -> int:
    """判断DataFrame中最后一根已完成bar的iloc索引

    对于日线：如果最后一根bar是今天且未收盘(15:00)，则取iloc[-2]；
    否则取iloc[-1]。

    Args:
        df: K线DataFrame（index为datetime）
        freq: 周期，目前仅支持 "d"
    Returns:
        最后一根已完成bar的iloc索引（-1 或 -2）
    """
    if df.empty:
        return -1

    now = datetime.now()
    last_ts = df.index[-1]

    if freq == "d":
        if last_ts.date() == now.date():
            if last_ts.hour == 15 and last_ts.minute == 0:
                return -1
            elif now.time() < time(15, 5):
                return -2 if len(df) >= 2 else -1
        elif last_ts.date() < now.date():
            return -1

    return -1


def fetch_all_kline(ts_codes: List[str], freq: str, bars: int = 500,
                    max_time: datetime = None) -> Dict[str, pd.DataFrame]:
    """批量获取多只股票的K线数据，使用全局 pytdx 连接

    Args:
        ts_codes: 股票代码列表
        freq: 周期（d/60m/15m）
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
                    # 日线/周线做前复权，确保历史价格与当前价格在同一坐标系
                    if freq in ('d', 'w'):
                        from datasource.adj_factor import apply_adj_factor
                        kline = apply_adj_factor(kline, ts_code, freq=freq)
                    result[ts_code] = kline
        except Exception as e:
            logger.warning(f"获取 {ts_code} {freq} K线失败: {e}")

    return result


# ---------------------------------------------------------------------------
# 自选股数据
# ---------------------------------------------------------------------------

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
        ts_codes: 纯6位股票代码列表
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _fmt_pct(val: Optional[float]) -> str:
    """格式化百分比，带正负号"""
    if val is None:
        return "-"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _fmt_price(val: Optional[float]) -> str:
    """格式化价格"""
    if val is None:
        return "-"
    return f"{val:.2f}"


def _calc_deviation_pct(price: float, boundary: float) -> Optional[float]:
    """计算价格偏离边界的百分比

    正值=价格在边界上方，负值=价格在边界下方。
    """
    if boundary is None or boundary == 0:
        return None
    return round((price - boundary) / boundary * 100, 2)


def _get_max_severity(signals: List[dict]) -> str:
    """获取信号列表中最严重的级别（danger > warn > info）"""
    severity_order = {"danger": 3, "warn": 2, "info": 1}
    max_sev = "info"
    for sig in signals:
        sev = EVENT_COLOR_MAP.get(sig["trigger_type"], {}).get("severity", "info")
        if severity_order.get(sev, 0) > severity_order.get(max_sev, 0):
            max_sev = sev
    return max_sev


# ---------------------------------------------------------------------------
# 布林带信号检测
# ---------------------------------------------------------------------------

def compute_bb_snapshot(df: pd.DataFrame, freq: str = "d") -> Optional[dict]:
    """计算单只股票的布林带快照（基于上一根已完成bar）

    Args:
        df: K线数据（需包含 open, high, low, close）
        freq: 周期，用于判断已完成bar
    Returns:
        快照字典，数据不足时返回 None
    """
    if len(df) < BB_WIN + 5:
        return None

    try:
        bb_mid, bb_upper, bb_lower = bollinger(df, BB_WIN, BB_K)
    except Exception as e:
        logger.warning(f"bollinger 计算失败: {e}")
        return None

    # 附加到临时df以获取已完成bar
    tmp = df.copy()
    tmp["bb_mid"] = bb_mid
    tmp["bb_upper"] = bb_upper
    tmp["bb_lower"] = bb_lower

    bar_idx = _get_completed_bar_index(tmp, freq)
    latest = tmp.iloc[bar_idx]

    price = float(latest.get("close", np.nan))
    mid_val = float(latest.get("bb_mid", np.nan)) if pd.notna(latest.get("bb_mid")) else None
    upper_val = float(latest.get("bb_upper", np.nan)) if pd.notna(latest.get("bb_upper")) else None
    lower_val = float(latest.get("bb_lower", np.nan)) if pd.notna(latest.get("bb_lower")) else None

    bb_width = None
    bb_pos = None
    if mid_val and mid_val > 0 and upper_val is not None and lower_val is not None:
        bb_width = round((upper_val - lower_val) / mid_val, 4)
        denom = upper_val - lower_val
        if denom > 0:
            bb_pos = round((price - lower_val) / denom, 3)

    return {
        "price": price,
        "bb_mid": mid_val,
        "bb_upper": upper_val,
        "bb_lower": lower_val,
        "bb_width": bb_width,
        "bb_pos": bb_pos,
    }


def detect_bb_signals(daily_df: pd.DataFrame, m1_df: pd.DataFrame = None,
                      freq: str = "d") -> List[dict]:
    """检测布林带穿越信号

    布林带参考线（upper/mid/lower）基于日线上一根已完成bar计算，
    穿越判断使用1分钟K线（前一根收盘价 vs 当前收盘价）。
    无1分钟数据时不触发（无法判断穿越方向）。

    - 上轨穿越：前一根收盘 < 参考上轨 <= 当前收盘（从下方穿越上来）
    - 中轨穿越：前一根收盘和当前收盘分列中轨两侧
    - 下轨穿越：前一根收盘 > 参考下轨 >= 当前收盘（从上方穿越下来）

    Args:
        daily_df: 日线K线数据（用于计算布林带参考线）
        m1_df: 1分钟K线数据（用于穿越判断），None或不足2根时不触发
        freq: 周期
    Returns:
        触发信号列表
    """
    if len(daily_df) < BB_WIN + 5:
        return []

    try:
        bb_mid, bb_upper, bb_lower = bollinger(daily_df, BB_WIN, BB_K)
    except Exception as e:
        logger.warning(f"bollinger 计算失败: {e}")
        return []

    tmp = daily_df.copy()
    tmp["bb_mid"] = bb_mid
    tmp["bb_upper"] = bb_upper
    tmp["bb_lower"] = bb_lower

    if len(tmp) < 2:
        return []

    # 上一根已完成bar（用于获取参考线）
    if freq == "d":
        ref = tmp.iloc[-2]
    else:
        ref_idx = _get_completed_bar_index(tmp, freq)
        ref = tmp.iloc[ref_idx]

    ref_upper = float(ref["bb_upper"]) if pd.notna(ref["bb_upper"]) else None
    ref_mid = float(ref["bb_mid"]) if pd.notna(ref["bb_mid"]) else None
    ref_lower = float(ref["bb_lower"]) if pd.notna(ref["bb_lower"]) else None

    # 获取1分钟K线的收盘价（用于穿越判断）
    if m1_df is None or len(m1_df) < 2:
        # 无1分钟数据时无法判断穿越方向，不触发
        return []

    prev_close = float(m1_df.iloc[-2]["close"])
    cur_close_val = float(m1_df.iloc[-1]["close"])

    base_signal = {
        "price": cur_close_val,
        "bb_mid": ref_mid,
        "bb_upper": ref_upper,
        "bb_lower": ref_lower,
        "freq": freq,
    }

    signals = []

    # 上轨穿越：从下方穿越到上方（prev_close < ref_upper <= cur_close）
    if ref_upper is not None and prev_close < ref_upper <= cur_close_val:
        dev_pct = _calc_deviation_pct(cur_close_val, ref_upper)
        signals.append(dict(base_signal, trigger_type=BB_UPPER_TOUCH,
                            trigger_label=TRIGGER_LABEL[BB_UPPER_TOUCH],
                            boundary=ref_upper, dev_pct=dev_pct))

    # 中轨穿越：从一侧穿越到另一侧
    if ref_mid is not None:
        mid_cross = (prev_close <= ref_mid < cur_close_val) or (cur_close_val <= ref_mid < prev_close)
        if mid_cross:
            dev_pct = _calc_deviation_pct(cur_close_val, ref_mid)
            signals.append(dict(base_signal, trigger_type=BB_MID_TOUCH,
                                trigger_label=TRIGGER_LABEL[BB_MID_TOUCH],
                                boundary=ref_mid, dev_pct=dev_pct))

    # 下轨穿越：从上方穿越到下方（prev_close > ref_lower >= cur_close）
    if ref_lower is not None and prev_close > ref_lower >= cur_close_val:
        dev_pct = _calc_deviation_pct(cur_close_val, ref_lower)
        signals.append(dict(base_signal, trigger_type=BB_LOWER_TOUCH,
                            trigger_label=TRIGGER_LABEL[BB_LOWER_TOUCH],
                            boundary=ref_lower, dev_pct=dev_pct))

    return signals


# ---------------------------------------------------------------------------
# 节点集群信号检测
# ---------------------------------------------------------------------------

def compute_node_cluster_prices(result: VolumeProfileResult) -> List[float]:
    """从 VolumeProfileResult 提取节点集群对应的价格列表

    使用 profile_df 中 is_peak=True 的行，取 price_mid。
    与新模块默认 peaks_show="peaks" 保持一致。

    Args:
        result: compute_volume_profile() 返回的 VolumeProfileResult
    Returns:
        去重排序后的价格列表
    """
    if result.profile_df is None or result.profile_df.empty:
        return []

    peak_rows = result.profile_df[result.profile_df["is_peak"]]
    if peak_rows.empty:
        return []

    prices = []
    for _, row in peak_rows.iterrows():
        p = float(row["price_mid"])
        if np.isfinite(p):
            prices.append(round(p, 4))

    return sorted(set(prices))


def detect_node_cluster_signals(m1_df: pd.DataFrame, profile: VolumeProfileResult,
                                freq: str = "d") -> List[dict]:
    """检测节点集群穿越信号

    从 VolumeProfileResult 提取筹码峰价格，用1分钟K线判断穿越：
    前一根收盘在峰价一侧，当前收盘穿越到另一侧。
    无1分钟数据或不足2根时不触发。

    Args:
        m1_df: 1分钟K线数据（用于穿越判断），None或不足2根时返回空列表
        profile: compute_volume_profile() 返回的 VolumeProfileResult
        freq: 周期
    Returns:
        触发信号列表
    """
    cluster_prices = compute_node_cluster_prices(profile)
    if not cluster_prices:
        return []

    if m1_df is None or len(m1_df) < 2:
        return []

    prev_close = float(m1_df.iloc[-2]["close"])
    cur_close_val = float(m1_df.iloc[-1]["close"])

    signals = []
    for cp in cluster_prices:
        # 穿越判断：前一根收盘在一侧，当前收盘在另一侧
        peak_cross = (prev_close <= cp < cur_close_val) or (cur_close_val <= cp < prev_close)
        if peak_cross:
            dev_pct = _calc_deviation_pct(cur_close_val, cp)
            signals.append({
                "trigger_type": NODE_CLUSTER_TOUCH,
                "trigger_label": TRIGGER_LABEL[NODE_CLUSTER_TOUCH],
                "price": cur_close_val,
                "cluster_price": cp,
                "boundary": cp,
                "dev_pct": dev_pct,
                "freq": freq,
            })

    return signals


def _format_volume(vol: float) -> str:
    """格式化成交量：自动选择万/亿单位"""
    if abs(vol) >= 1e8:
        return f"{vol / 1e8:.1f}亿"
    elif abs(vol) >= 1e4:
        return f"{vol / 1e4:.1f}万"
    else:
        return f"{vol:.0f}"


# ---------------------------------------------------------------------------
# 行情图片渲染
# ---------------------------------------------------------------------------

def render_monitoring_chart(
    df: pd.DataFrame,
    bb_mid: pd.Series,
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    profile: Optional[VolumeProfileResult],
    ts_code: str,
    stock_name: str,
    out_png: str = None,
    bars_to_plot: int = 120,
    trigger_bar_idx: Optional[int] = None,
) -> Optional[str]:
    """渲染布林带+节点集群K线图并导出PNG

    生成单行子图：K线+布林带+节点集群色带+POC/VAH/VAL水平线。
    仅绘制最近 bars_to_plot 根bar，优化移动端可读性。

    Args:
        df: K线DataFrame（index为datetime）
        bb_mid/bb_upper/bb_lower: 布林带序列（与df同长度）
        profile: VolumeProfileResult（可选，为None时不绘制节点集群）
        ts_code: 股票代码
        stock_name: 股票名称
        out_png: 输出PNG路径，None则使用临时文件
        bars_to_plot: 绘制最近多少根bar
        trigger_bar_idx: 触发信号的bar在原始df中的iloc索引，用于标记竖线
    Returns:
        PNG文件路径，失败返回None
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly 未安装，跳过图表渲染")
        return None

    if df.empty:
        return None

    # 截取绘制范围
    n_tail = min(bars_to_plot, len(df))
    df_plot = df.tail(n_tail).copy()
    bb_mid_plot = bb_mid.iloc[-n_tail:].values if len(bb_mid) >= n_tail else bb_mid.tail(n_tail).values
    bb_upper_plot = bb_upper.iloc[-n_tail:].values if len(bb_upper) >= n_tail else bb_upper.tail(n_tail).values
    bb_lower_plot = bb_lower.iloc[-n_tail:].values if len(bb_lower) >= n_tail else bb_lower.tail(n_tail).values

    if len(df_plot) < 5:
        return None

    x = np.arange(len(df_plot), dtype=float)
    tick_text = [ts.strftime("%m-%d") for ts in df_plot.index]

    title = f"日线 | {stock_name} {ts_code} | 筹码峰已标注"

    fig = go.Figure()

    # 计算右侧 profile 区域锚点（用于筹码分布柱状图）
    # 与 luxalgo_volume_profile_pytdx.py HTML 渲染保持一致
    profile_anchor = None
    if profile is not None and not profile.profile_df.empty:
        # profile_width = 0.31，offset = 13
        profile_width_bars = max(1.0, len(df_plot) * 0.31)
        offset = int(profile_width_bars) + 13
        last_x = len(df_plot) - 1
        profile_anchor = last_x + offset

    # K线
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df_plot["open"], high=df_plot["high"],
            low=df_plot["low"], close=df_plot["close"],
            name="K线",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        ),
    )

    # 筹码分布柱状图（参考 luxalgo_volume_profile HTML 渲染）
    # 在每个价位画两个矩形：bullish(蓝/灰) + down(黄/浅灰)
    # 长度 = 该价位成交量 / 最大成交量 * profile_width_bars
    if profile_anchor is not None:
        max_vol = float(profile.profile_df["total_volume"].max()) or 1.0
        profile_width_bars = max(1.0, len(df_plot) * 0.31)
        peak_data = []  # 收集 peak 行数据，用于绘制迷你多空柱
        for _, row in profile.profile_df.iterrows():
            total = float(row["total_volume"])
            bull = float(row["bullish_volume"])
            down = max(total - bull, 0.0)
            y0 = float(row["price_low"]) + 0.1 * profile.price_step
            y1 = float(row["price_low"]) + 0.9 * profile.price_step

            in_va = bool(row["is_value_area"])
            up_base = "rgba(41,98,255,0.70)" if in_va else "rgba(93,96,107,0.50)"
            down_base = "rgba(251,192,45,0.70)" if in_va else "rgba(209,212,220,0.50)"

            bull_w = bull / max_vol * profile_width_bars
            down_w = down / max_vol * profile_width_bars

            # bullish 部分（从锚点向左延伸）
            x1 = profile_anchor
            x0 = x1 - bull_w
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=x0, x1=x1, y0=y0, y1=y1,
                          line=dict(width=0), fillcolor=up_base, layer="above")
            # down 部分（向左接续）
            x1d = x0
            x0d = x1d - down_w
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=x0d, x1=x1d, y0=y0, y1=y1,
                          line=dict(width=0), fillcolor=down_base, layer="above")
            end_x = x0d

            # peak 节点高亮：叠加蓝色色带覆盖整个左半部分
            if bool(row["is_peak"]):
                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=0, x1=end_x, y0=y0, y1=y1,
                              line=dict(width=0), fillcolor="rgba(33,150,243,0.50)",
                              layer="above")
                # 收集 peak 行数据，稍后统一绘制迷你多空柱
                peak_data.append({
                    "y0": y0, "y1": y1, "end_x": end_x,
                    "bullish_volume": float(row["bullish_volume"]),
                    "bearish_volume": float(row["bearish_volume"]),
                })

    # 筹码峰迷你多空柱（在 peak 色带内部绘制绿色多头+红色空头水平柱）
    if profile_anchor is not None and peak_data:
        max_peak_vol = max(max(p["bullish_volume"], p["bearish_volume"]) for p in peak_data) or 1.0
        mini_max_w = profile_width_bars * 0.6
        bar_h_ratio = 0.4
        for pd_item in peak_data:
            y_range = pd_item["y1"] - pd_item["y0"]
            bar_y0 = pd_item["y0"] + y_range * (0.5 - bar_h_ratio / 2)
            bar_y1 = pd_item["y0"] + y_range * (0.5 + bar_h_ratio / 2)
            # 多头柱（绿色，从左端向右）
            bull_w = pd_item["bullish_volume"] / max_peak_vol * mini_max_w
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=0, x1=bull_w, y0=bar_y0, y1=bar_y1,
                          line=dict(width=0), fillcolor="rgba(38,166,154,0.85)",
                          layer="above")
            # 空头柱（红色，紧接多头柱右侧）
            bear_w = pd_item["bearish_volume"] / max_peak_vol * mini_max_w
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=bull_w, x1=bull_w + bear_w, y0=bar_y0, y1=bar_y1,
                          line=dict(width=0), fillcolor="rgba(239,83,80,0.85)",
                          layer="above")

    # 布林带填充区域
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([bb_upper_plot, bb_lower_plot[::-1]]),
            fill="toself",
            fillcolor="rgba(33,150,243,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            mode="lines", showlegend=False, hoverinfo="skip",
            name="BB区域",
        ),
    )

    # 布林带上轨
    fig.add_trace(
        go.Scatter(x=x, y=bb_upper_plot, mode="lines",
                   line=dict(color="#2196F3", width=1.2, dash="dash"),
                   name="BB上轨"),
    )

    # 布林带中轨
    fig.add_trace(
        go.Scatter(x=x, y=bb_mid_plot, mode="lines",
                   line=dict(color="#FF9800", width=1.5),
                   name="BB中轨"),
    )

    # 布林带下轨
    fig.add_trace(
        go.Scatter(x=x, y=bb_lower_plot, mode="lines",
                   line=dict(color="#2196F3", width=1.2, dash="dash"),
                   name="BB下轨"),
    )

    # 筹码峰价格标签 + 多空量标签（右侧标注）
    if profile_anchor is not None:
        peak_rows = profile.profile_df[profile.profile_df["is_peak"]]
        label_x = profile_anchor + 1
        for _, row in peak_rows.iterrows():
            price_mid = float(row["price_mid"])
            # 价格标签
            fig.add_annotation(
                x=label_x, y=price_mid,
                text=f"峰 {price_mid:.2f}",
                showarrow=False, xanchor="left",
                font=dict(color="#2196F3", size=11),
                bgcolor="rgba(19,23,34,0.85)",
            )
            # 多空量标签
            bull_vol = float(row["bullish_volume"])
            bear_vol = float(row["bearish_volume"])
            vol_text = f"多{_format_volume(bull_vol)} 空{_format_volume(bear_vol)}"
            fig.add_annotation(
                x=label_x, y=price_mid,
                text=vol_text,
                showarrow=False, xanchor="left",
                font=dict(color="#d1d4dc", size=9),
                bgcolor="rgba(19,23,34,0.85)",
                yshift=-14,
            )

    # X轴范围：容纳 profile 柱状图
    if profile_anchor is not None:
        fig.update_xaxes(range=[-2, profile_anchor + 4])

    # 已完成bar竖线标记
    bar_idx = _get_completed_bar_index(df, "d")
    if bar_idx == -2 and len(df_plot) >= 2:
        completed_x = x[-2]
        fig.add_vline(x=completed_x, line_width=1, line_dash="dash",
                      line_color="rgba(255,255,255,0.4)")

    # 触发信号bar竖线标记（测试模式用）
    if trigger_bar_idx is not None:
        # 计算在截取后的df_plot中的位置
        n_tail = min(bars_to_plot, len(df))
        start_idx = len(df) - n_tail
        rel_idx = trigger_bar_idx - start_idx
        if 0 <= rel_idx < len(x):
            fig.add_vline(x=x[rel_idx], line_width=2, line_dash="solid",
                          line_color="#F44336",
                          annotation_text="触发点", annotation_position="top right",
                          annotation_font_color="#F44336", annotation_font_size=11)

    # X轴标签
    step = max(1, len(df_plot) // 8)
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x[::step]),
        ticktext=tick_text[::step],
        rangeslider_visible=False,
    )

    fig.update_layout(
        template="plotly_dark",
        height=640,
        width=1200,
        hovermode="x unified",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        title=dict(text=title, x=0.01, font=dict(size=14, color="#d1d4dc")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=120, t=60, b=40),
    )

    # 导出PNG
    if out_png is None:
        fd, out_png = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    try:
        fig.write_image(out_png, width=1200, height=640, scale=2)
        return out_png
    except Exception as e:
        print(f"图表PNG导出失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 飞书通知
# ---------------------------------------------------------------------------

def send_card_notification(header_title: str, header_template: str, elements: list):
    """发送飞书卡片通知

    Args:
        header_title: 卡片标题
        header_template: header颜色模板
        elements: 卡片内容元素列表
    """
    try:
        _notifier.send_card(header_title=header_title, header_template=header_template,
                            elements=elements)
    except Exception as e:
        print(f"飞书卡片通知发送失败: {e}")
        raise


def send_chart_image(png_path: str):
    """发送飞书图片消息

    Args:
        png_path: PNG文件路径
    """
    try:
        _notifier.send_image(png_path)
    except Exception as e:
        print(f"飞书图片发送失败: {e}")
        raise


# ---------------------------------------------------------------------------
# 飞书卡片生成
# ---------------------------------------------------------------------------

def generate_monitoring_card(all_stocks: List[Dict],
                             triggered_stocks: List[Dict]) -> dict:
    """生成布林带+节点集群监控飞书卡片消息

    Args:
        all_stocks: 全部自选股列表
        triggered_stocks: 触发信号的股票列表
    Returns:
        飞书卡片参数 dict
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 全局最严重级别决定header颜色
    all_signals = []
    for item in triggered_stocks:
        all_signals.extend(item["signals"])
    max_sev = _get_max_severity(all_signals) if all_signals else "info"
    header_template = SEVERITY_TEMPLATE.get(max_sev, "blue")

    # 概览统计
    trigger_counts = {BB_UPPER_TOUCH: 0, BB_MID_TOUCH: 0, BB_LOWER_TOUCH: 0, NODE_CLUSTER_TOUCH: 0}
    for item in triggered_stocks:
        for sig in item["signals"]:
            tt = sig["trigger_type"]
            if tt in trigger_counts:
                trigger_counts[tt] += 1

    elements = []

    # 概览行
    overview = (
        f"自选股 {len(all_stocks)} 只 | 触发 {len(triggered_stocks)} 只\n"
        f"上轨 {trigger_counts[BB_UPPER_TOUCH]} | "
        f"中轨 {trigger_counts[BB_MID_TOUCH]} | "
        f"下轨 {trigger_counts[BB_LOWER_TOUCH]} | "
        f"节点 {trigger_counts[NODE_CLUSTER_TOUCH]}"
    )
    elements.append({"tag": "markdown", "content": overview})

    for idx, item in enumerate(triggered_stocks):
        stock = item["stock"]
        signals = item["signals"]
        name = stock.get('stock_name', '-')
        code = stock.get('ts_code', '-')
        priority = stock.get('priority', '')
        score = stock.get('weighted_score', 0)
        change = stock.get('change_pct', 0)
        cap = stock.get('total_market_cap', 0)

        change_str = f"+{change}" if change > 0 else str(change)

        # 分隔线
        if idx > 0:
            elements.append({"tag": "hr"})

        # 股票标题
        title_md = f"**{name} {code}**  {priority}  {score}分\n涨跌 {change_str}%  市值 {cap:.0f}亿"
        elements.append({"tag": "markdown", "content": title_md})

        # 触发信号详情
        for sig in signals:
            emoji = TRIGGER_EMOJI.get(sig["trigger_type"], "📌")
            trigger_text = sig["trigger_label"]
            price = sig.get("price", 0)

            sig_lines = [f"{emoji} {trigger_text}"]
            sig_lines.append(f"  现价: {_fmt_price(price)}")

            boundary = sig.get("boundary")
            dev_pct = sig.get("dev_pct")
            if boundary is not None:
                boundary_label = {
                    BB_UPPER_TOUCH: "上轨",
                    BB_MID_TOUCH: "中轨",
                    BB_LOWER_TOUCH: "下轨",
                    NODE_CLUSTER_TOUCH: "节点",
                }.get(sig["trigger_type"], "边界")
                dev_str = _fmt_pct(dev_pct)
                sig_lines.append(f"  {boundary_label}: {_fmt_price(boundary)}  偏离: {dev_str}")

            # 布林带上下文
            if sig["trigger_type"] in (BB_UPPER_TOUCH, BB_MID_TOUCH, BB_LOWER_TOUCH):
                if sig.get("bb_upper") is not None:
                    sig_lines.append(f"  上轨: {_fmt_price(sig['bb_upper'])}")
                if sig.get("bb_mid") is not None:
                    sig_lines.append(f"  中轨: {_fmt_price(sig['bb_mid'])}")
                if sig.get("bb_lower") is not None:
                    sig_lines.append(f"  下轨: {_fmt_price(sig['bb_lower'])}")

            elements.append({"tag": "markdown", "content": "\n".join(sig_lines)})

        # 布林带快照
        snapshot = item.get("bb_snapshot")
        if snapshot:
            snap_lines = [f"  BB: 上{_fmt_price(snapshot.get('bb_upper'))} "
                          f"中{_fmt_price(snapshot.get('bb_mid'))} "
                          f"下{_fmt_price(snapshot.get('bb_lower'))}"]
            if snapshot.get("bb_width") is not None:
                snap_lines.append(f"  宽度: {snapshot['bb_width']:.4f}  位置: {snapshot.get('bb_pos', '-')}")
            elements.append({"tag": "markdown", "content": "\n".join(snap_lines)})

    return {
        "header_title": f"BB+节点监控 {now_str}",
        "header_template": header_template,
        "elements": elements,
    }


# ---------------------------------------------------------------------------
# 主监控流程
# ---------------------------------------------------------------------------

def run_monitor(dry_run: bool = False):
    """执行布林带+节点集群自选股监控任务

    流程：获取自选股 → 拉取日线+15分钟K线 → 检测布林带信号+节点集群信号 →
          发送卡片消息+行情图片

    Args:
        dry_run: True时仅打印检测结果，不发送通知
    """
    logger.info("开始 布林带+节点集群 自选股监控")

    # 1. 获取自选股列表
    with get_session() as session:
        stocks = get_monitored_stocks(session)

    if not stocks:
        logger.warning("无自选股")
        return

    logger.info(f"自选股数量: {len(stocks)}")
    ts_codes = [s["ts_code"] for s in stocks]

    # 2. 计算当日涨跌幅
    logger.info("获取日K线计算涨跌幅...")
    change_map = compute_daily_change_pct(ts_codes)
    for s in stocks:
        s['change_pct'] = change_map.get(s['ts_code'], 0.0)

    # 3. 拉取日线K线数据
    now = datetime.now().replace(second=0, microsecond=0)
    logger.info("开始获取日线行情数据...")
    daily_data = fetch_all_kline(ts_codes, 'd', bars=250,
                                 max_time=now + pd.Timedelta(days=1))
    logger.info(f"日线获取完成: {len(daily_data)}/{len(ts_codes)} 只股票")

    # 4. 拉取15分钟K线数据（用于volume profile计算）
    logger.info("开始获取15分钟行情数据...")
    ltf_data = fetch_all_kline(ts_codes, '15m', bars=8000,
                                max_time=now + pd.Timedelta(minutes=15))
    logger.info(f"15分钟线获取完成: {len(ltf_data)}/{len(ts_codes)} 只股票")

    # 5. 拉取1分钟K线数据（用于盘中触及检测）
    logger.info("开始获取1分钟行情数据...")
    m1_data = fetch_all_kline(ts_codes, '1m', bars=2,
                               max_time=now + pd.Timedelta(minutes=1))
    logger.info(f"1分钟线获取完成: {len(m1_data)}/{len(ts_codes)} 只股票")

    # 6. 逐只检测信号
    triggered_stocks = []
    for stock in stocks:
        ts_code = stock["ts_code"]
        all_signals = []

        daily_df = daily_data.get(ts_code)
        if daily_df is None or daily_df.empty:
            continue

        # 6a. 布林带信号检测
        m1_df = m1_data.get(ts_code)
        bb_signals = detect_bb_signals(daily_df, m1_df=m1_df, freq="d")
        all_signals.extend(bb_signals)

        # 6b. 节点集群信号检测
        ltf_df = ltf_data.get(ts_code)
        node_signals = []
        profile = None
        if ltf_df is not None and not ltf_df.empty and len(daily_df) >= 2:
            try:
                # compute_volume_profile 需要 DataFrame 带 datetime 列（非索引）
                ltf_for_vp = ltf_df.reset_index()

                # reset_index 后列名可能是 "index" 而非 "datetime"
                if "datetime" not in ltf_for_vp.columns:
                    for col in ["index", "date", "time"]:
                        if col in ltf_for_vp.columns:
                            ltf_for_vp = ltf_for_vp.rename(columns={col: "datetime"})
                            break

                if "datetime" not in ltf_for_vp.columns:
                    raise ValueError("无法确定 datetime 列")

                vp_cfg = VolumeProfileConfig(
                    profile_lookback_length=len(ltf_for_vp),
                    profile_number_of_rows=VP_ROWS,
                    value_area_threshold=VP_VALUE_AREA_PCT,
                    peaks_show="peaks",
                    peaks_detection_percent=VP_PEAK_DETECTION_PCT,
                    volume_node_threshold=VP_NODE_THRESHOLD_PCT,
                )
                profile = compute_volume_profile(ltf_for_vp, cfg=vp_cfg)
                node_signals = detect_node_cluster_signals(m1_df, profile, freq="d")
                all_signals.extend(node_signals)
            except Exception as e:
                logger.warning(f"volume_profile 计算失败 {ts_code}: {e}")

        if not all_signals:
            continue

        # 收集布林带快照
        bb_snapshot = compute_bb_snapshot(daily_df, freq="d")

        item = {
            'stock': stock,
            'signals': all_signals,
            'bb_snapshot': bb_snapshot,
            'daily_df': daily_df,
            'profile': profile,
        }
        triggered_stocks.append(item)

    logger.info(f"检测到 {len(triggered_stocks)} 只股票有触发信号")

    # 6. 发送通知（冷却过滤）
    if triggered_stocks:
        # 过滤冷却期内的信号
        for item in triggered_stocks:
            ts_code = item["stock"]["ts_code"]
            item["signals"] = [
                sig for sig in item["signals"]
                if _should_notify(ts_code, sig["trigger_type"], sig.get("boundary", 0))
            ]
        # 移除所有信号都被冷却的股票
        triggered_stocks = [item for item in triggered_stocks if item["signals"]]

    if triggered_stocks:
        if dry_run:
            print("[dry-run] 不发送通知，信号详情：")
            for item in triggered_stocks:
                stock = item["stock"]
                print(f"  {stock['stock_name']} {stock['ts_code']}:")
                for sig in item["signals"]:
                    print(f"    {sig['trigger_label']} 现价={sig['price']:.2f} "
                          f"边界={sig.get('boundary', '-')}")
        else:
            # 6a. 发送卡片消息
            card = generate_monitoring_card(stocks, triggered_stocks)
            logger.info("发送监控卡片...")
            send_card_notification(
                header_title=card["header_title"],
                header_template=card["header_template"],
                elements=card["elements"],
            )

            # 6b. 为每只触发股票渲染并发送行情图片
            for item in triggered_stocks:
                stock = item["stock"]
                ts_code = stock["ts_code"]
                stock_name = stock["stock_name"]
                daily_df = item["daily_df"]
                profile = item.get("profile")

                # 计算布林带序列（用于图表渲染）
                try:
                    bb_mid_s, bb_upper_s, bb_lower_s = bollinger(daily_df, BB_WIN, BB_K)
                except Exception as e:
                    logger.warning(f"bollinger 计算失败 {ts_code}: {e}")
                    continue

                png_path = render_monitoring_chart(
                    df=daily_df,
                    bb_mid=bb_mid_s,
                    bb_upper=bb_upper_s,
                    bb_lower=bb_lower_s,
                    profile=profile,
                    ts_code=ts_code,
                    stock_name=stock_name,
                )
                if png_path and os.path.exists(png_path):
                    try:
                        logger.info(f"发送 {stock_name} 行情图片...")
                        send_chart_image(png_path)
                    finally:
                        try:
                            os.unlink(png_path)
                        except OSError:
                            pass
    else:
        logger.info("无触发信号，不推送")

    logger.info("检测完成")
    close_api()


# ---------------------------------------------------------------------------
# 持续监控调度
# ---------------------------------------------------------------------------

_MONITOR_LOCK_FILE = "/tmp/bb_node_monitor.lock"


def start_scheduled_monitor():
    """启动持续监控（在交易时段内循环执行）

    交易时段：
    - 上午：9:30 - 11:30
    - 下午：13:00 - 15:00

    每轮执行 run_monitor()，完成后立即开始下一轮，不设最小间隔。
    午休（11:30-13:00）自动暂停，收盘（15:00）自动退出。
    文件锁互斥防止重复启动。

    调用方式：
        python app/monitoring.py --schedule
    """
    import fcntl
    from datasource.trade_calendar import is_trading_day

    # 交易日检查
    if not is_trading_day():
        logger.info("今天不是交易日，监控不启动")
        return

    # 文件锁互斥：防止重复启动
    lock_fd = open(_MONITOR_LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        logger.warning("另一个监控进程已在运行，退出")
        lock_fd.close()
        return

    logger.info("监控系统启动，等待交易时段...")

    try:
        while True:
            now = datetime.now()
            current_time = now.time()

            # 交易时段判断
            morning_start = time(9, 30)
            morning_end = time(11, 30)
            afternoon_start = time(13, 0)
            afternoon_end = time(15, 0)

            in_morning = morning_start <= current_time <= morning_end
            in_afternoon = afternoon_start <= current_time <= afternoon_end

            if in_morning or in_afternoon:
                # 交易时段：执行监控
                try:
                    run_monitor()
                except Exception as e:
                    logger.error(f"监控执行失败：{e}", exc_info=True)
                # 一轮完成后等待30秒再开始下一轮
                import time as _time
                _time.sleep(30)
                continue

            if current_time > afternoon_end:
                # 收盘后退出
                logger.info("已收盘，监控系统退出")
                break

            if morning_end < current_time < afternoon_start:
                # 午休等待
                logger.info("午休中，等待 13:00 恢复...")
                wait_seconds = (
                    datetime(now.year, now.month, now.day, 13, 0) - now
                ).total_seconds()
                if wait_seconds > 0:
                    import time as _time
                    _time.sleep(min(wait_seconds, 60))
                continue

            if current_time < morning_start:
                # 开盘前等待
                logger.info("等待开盘（9:30）...")
                import time as _time
                _time.sleep(30)
                continue
    finally:
        # 释放文件锁
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
        except Exception:
            pass
        # 清理锁文件
        try:
            os.unlink(_MONITOR_LOCK_FILE)
        except OSError:
            pass

    logger.info("监控系统已停止")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="布林带+节点集群 自选股监控")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印检测结果，不发送通知",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：遍历历史数据，检测自选股中最后一次触发信号的时间点，渲染并推送图片",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="启动持续监控调度器（交易时段内循环执行）",
    )
    return parser.parse_args()


def test_last_triggered():
    """测试入口：遍历历史数据，检测自选股中最后一次触发信号的时间点，渲染并推送图片

    流程：获取自选股 → 拉取日线+15分钟K线 → 逐bar检测信号 →
          找出每只股票最后一次触发的bar → 渲染图片并推送
    """
    print(f"[{datetime.now()}] 开始测试模式：检测最后一次触发信号")

    # 1. 获取自选股列表
    with get_session() as session:
        stocks = get_monitored_stocks(session)

    if not stocks:
        print("无自选股")
        return

    print(f"自选股数量: {len(stocks)}")
    ts_codes = [s["ts_code"] for s in stocks]

    # 2. 拉取日线K线数据
    now = datetime.now().replace(second=0, microsecond=0)
    print("开始获取日线行情数据...")
    daily_data = fetch_all_kline(ts_codes, 'd', bars=250,
                                 max_time=now + pd.Timedelta(days=1))
    print(f"日线获取完成: {len(daily_data)}/{len(ts_codes)} 只股票")

    # 3. 拉取15分钟K线数据（用于volume profile计算）
    print("开始获取15分钟行情数据...")
    ltf_data = fetch_all_kline(ts_codes, '15m', bars=8000,
                                max_time=now + pd.Timedelta(minutes=15))
    print(f"15分钟线获取完成: {len(ltf_data)}/{len(ts_codes)} 只股票")

    # 4. 逐只股票逐bar检测信号
    last_triggered = []

    for stock in stocks:
        ts_code = stock["ts_code"]
        daily_df = daily_data.get(ts_code)
        if daily_df is None or daily_df.empty or len(daily_df) < BB_WIN + 5:
            continue

        # 排除最后一根未完成bar（避免信号闪烁）
        completed_idx = _get_completed_bar_index(daily_df, "d")
        if completed_idx == -2:
            # 最后一根是今天未完成，前一根是昨天已完成
            completed_len = len(daily_df) - 1
        else:
            completed_len = len(daily_df)

        # 4a. 逐bar检测布林带信号
        bb_mid, bb_upper, bb_lower = bollinger(daily_df, BB_WIN, BB_K)
        triggered_bars = []

        for i in range(BB_WIN, completed_len):
            bar = daily_df.iloc[i]
            cur_high = float(bar["high"])
            cur_low = float(bar["low"])
            cur_close = float(bar["close"])
            bar_time = daily_df.index[i]

            # 使用当前bar的前一根bar作为参考线
            ref_upper = float(bb_upper.iloc[i - 1])
            ref_mid = float(bb_mid.iloc[i - 1])
            ref_lower = float(bb_lower.iloc[i - 1])

            signals = []
            # 上轨触及：当前bar最高价 >= 参考上轨
            if cur_high >= ref_upper:
                signals.append(BB_UPPER_TOUCH)
            # 中轨触及：前一天在中轨下方（close < mid），今天在中轨上方（low <= mid），表示穿越中轨
            prev_close = float(daily_df.iloc[i - 1]["close"])
            if cur_low <= ref_mid <= prev_close:
                # 价格从下方穿越中轨到上方
                signals.append(BB_MID_TOUCH)
            elif cur_high >= ref_mid >= prev_close:
                # 价格从上方穿越中轨到下方
                signals.append(BB_MID_TOUCH)
            # 下轨触及：当前bar最低价 <= 参考下轨
            if cur_low <= ref_lower:
                signals.append(BB_LOWER_TOUCH)

            if signals:
                triggered_bars.append({
                    'time': bar_time,
                    'signals': signals,
                    'close': cur_close,
                    'high': cur_high,
                    'low': cur_low,
                })

        # 4b. 检测节点集群信号
        ltf_df = ltf_data.get(ts_code)
        node_triggered_bars = []
        profile = None
        if ltf_df is not None and not ltf_df.empty and len(daily_df) >= 2:
            try:
                ltf_for_vp = ltf_df.reset_index()
                if 'datetime' not in ltf_for_vp.columns:
                    for col in ["index", "date", "time"]:
                        if col in ltf_for_vp.columns:
                            ltf_for_vp = ltf_for_vp.rename(columns={col: "datetime"})
                            break
                if 'datetime' in ltf_for_vp.columns:
                    vp_cfg = VolumeProfileConfig(
                        profile_lookback_length=VP_LOOKBACK,
                        profile_number_of_rows=VP_ROWS,
                        value_area_threshold=VP_VALUE_AREA_PCT,
                        peaks_show="peaks",
                        peaks_detection_percent=VP_PEAK_DETECTION_PCT,
                        volume_node_threshold=VP_NODE_THRESHOLD_PCT,
                    )
                    profile = compute_volume_profile(ltf_for_vp, cfg=vp_cfg)
            except Exception as e:
                logger.warning(f"volume_profile 计算失败 {ts_code}: {e}")

        if profile:
            cluster_prices = compute_node_cluster_prices(profile)
            if cluster_prices:
                for i in range(1, completed_len):
                    bar = daily_df.iloc[i]
                    cur_high = float(bar["high"])
                    cur_low = float(bar["low"])
                    cur_close = float(bar["close"])
                    bar_time = daily_df.index[i]

                    touched_prices = []
                    for cp in cluster_prices:
                        if cur_low <= cp <= cur_high:
                            touched_prices.append(cp)

                    if touched_prices:
                        node_triggered_bars.append({
                            'time': bar_time,
                            'signals': [NODE_CLUSTER_TOUCH],
                            'close': cur_close,
                            'high': cur_high,
                            'low': cur_low,
                            'cluster_prices': touched_prices,
                        })

        # 4c. 合并两种信号
        all_triggered = []
        all_triggered.extend(triggered_bars)
        all_triggered.extend(node_triggered_bars)
        all_triggered.sort(key=lambda x: x['time'])

        if all_triggered:
            last = all_triggered[-1]
            last_triggered.append({
                'stock': stock,
                'last_bar': last,
                'total_triggers': len(all_triggered),
                'bb_triggers': len(triggered_bars),
                'node_triggers': len(node_triggered_bars),
                'profile': profile,
                'daily_df': daily_df,
                'bb_mid': bb_mid,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
            })
            print(f"  {stock['stock_name']} {ts_code}: 共触发 {len(all_triggered)} 次 "
                  f"(BB {len(triggered_bars)}, 峰 {len(node_triggered_bars)})，"
                  f"最后一次: {last['time']} 信号={last['signals']} 收盘={last['close']:.2f}")
            if NODE_CLUSTER_TOUCH in last['signals'] and last.get('cluster_prices'):
                print(f"    触发筹码峰价格: {last['cluster_prices']}")
        else:
            print(f"  {stock['stock_name']} {ts_code}: 无触发信号")

    # 5. 渲染并推送图片
    if not last_triggered:
        print("无任何股票触发信号")
        return

    # 按最后一次触发时间排序
    last_triggered.sort(key=lambda x: x['last_bar']['time'])

    # 渲染所有触发股票的图片
    png_paths = []
    for item in last_triggered:
        stock = item['stock']
        ts_code = stock['ts_code']
        stock_name = stock['stock_name']
        daily_df = item['daily_df']
        profile = item.get('profile')
        last_bar = item['last_bar']

        print(f"\n渲染 {stock_name} {ts_code} 图片（最后一次触发: {last_bar['time']}）...")

        # 找到最后一次触发bar在原始df中的iloc索引
        trigger_bar_idx = None
        try:
            trigger_bar_idx = daily_df.index.get_loc(last_bar['time'])
        except KeyError:
            pass

        png_path = render_monitoring_chart(
            df=daily_df,
            bb_mid=item['bb_mid'],
            bb_upper=item['bb_upper'],
            bb_lower=item['bb_lower'],
            profile=profile,
            ts_code=ts_code,
            stock_name=stock_name,
            trigger_bar_idx=trigger_bar_idx,
        )
        if png_path and os.path.exists(png_path):
            png_paths.append(png_path)
        else:
            print(f"  渲染失败")

    # 推送图片
    if png_paths:
        print(f"\n推送 {len(png_paths)} 张图片...")
        for png_path in png_paths:
            try:
                send_chart_image(png_path)
            except Exception as e:
                print(f"推送失败: {e}")
            finally:
                try:
                    os.unlink(png_path)
                except OSError:
                    pass
        print("推送完成")
    else:
        print("无图片可推送")

    print(f"[{datetime.now()}] 测试模式完成")
    close_api()


def main():
    args = parse_args()
    if args.test:
        test_last_triggered()
    elif args.schedule:
        start_scheduled_monitor()
    else:
        run_monitor(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
