#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AMP + 底背离 选股策略

策略目标：
  寻找"强趋势后第一次回踩"的个股，并用背离信号确认买入时机
  - 使用 AMP（自适应移动通道）识别强趋势和回踩位置
  - 使用背离引擎（MACD/Hist/OBV）确认底部背离信号

核心逻辑：
  1) 用 AMP 逐 bar 扫描，寻找"强趋势后第一次回踩"
  2) 用背离引擎确认最近 3 天是否出现底背离（常规底背离 + 隐藏底背离）
  3) 默认拉取 810 根日线，AMP slope 使用 adjusted-space slope（对数空间）口径
  4) 运行时先做"目标 bar 快速过滤"，只有通过后才进入历史逐 bar 扫描

条件群 A（当前 bar 的 AMP 条件）：
  - amp_strength >= 0.70（强趋势）
  - mid_slope > 0.001（上涨斜率为正且有一定强度）
  - close_pos <= 0.50（价格不在通道顶部，处于中轴及以下）

条件群 B（历史强趋势冲顶）：
  - 最近 lookback_trend bars（默认 100）内，存在"最近一次强趋势冲顶 bar"
  - 该 bar 满足：
    * amp_strength >= 0.70
    * close_pos >= 0.98（价格触及通道顶部）
    * mid_slope > 0.001（斜率为正且有一定强度）

条件群 C（回踩确认）：
  - 当前 bar 的 close_pos <= 0.50（处于回踩位置）

条件群 D（背离确认）：
  - 最近 lookback_div bars（默认 3）内，出现任意指标的底背离
  - 支持 Positive Regular（常规底背离）或 Positive Hidden（隐藏底背离）
  - 默认指标：MACD / Hist / OBV

核心指标说明：
  - amp_strength: AMP 趋势强度（Pearson R 相关系数，范围 -1 到 1，越接近 1 趋势越强）
  - mid_slope: 中轴斜率（adjusted-space slope，每 bar 的对数收益率）
  - close_pos: 收盘价在通道内的相对位置（0-1，0 表示在下轨，1 表示在上轨）
  - channel_width: 通道宽度（相对中轴的价格波动范围）

How to Run:
  # 基础用法（扫描全部股票）
  python selections/stock_picker.py --date 2026-03-07
  
  # 指定日期和输出文件
  python selections/stock_picker.py --date 2026-03-07 --output output/picks_2026-03-07.xlsx
  
  # 限制处理股票数（测试用）
  python selections/stock_picker.py --date 2026-03-07 --max-stocks 100
  
  # 调整 AMP 参数（放宽 mid_slope 范围）
  python selections/stock_picker.py --date 2026-03-07 --amp-min-mid-slope 0.0 --amp-max-mid-slope 0.03
  
  # 调整背离回看天数
  python selections/stock_picker.py --date 2026-03-07 --div-lookback-bars 5
  
  # 只使用 MACD 指标检测背离
  python selections/stock_picker.py --date 2026-03-07 --no-hist --no-obv
  
  # 查看帮助信息
  python selections/stock_picker.py --help

参数说明：
  --date: 目标日期（YYYY-MM-DD 格式），必选
  --output: 输出 Excel 文件路径，默认 output/selected_stocks_{日期}.xlsx
  --max-stocks: 最大处理股票数（测试用），默认无限制
  --bars: 每只股票拉取的 K 线数量，默认 810
  --amp-min-strength: AMP 强度最小值，默认 0.70
  --amp-min-mid-slope: mid_slope 最小值，默认 0.0
  --amp-max-mid-slope: mid_slope 最大值，默认 0.020
  --amp-pullback-close-pos-max: close_pos 最大值，默认 0.50
  --trend-peak-close-pos-min: 历史冲顶 close_pos 最小值，默认 0.98
  --trend-lookback-bars: 历史冲顶回看 bar 数，默认 100
  --trend-peak-min-mid-slope: 历史冲顶 mid_slope 最小值，默认 0.0
  --div-lookback-bars: 背离回看 bar 数，默认 3
  --no-macd: 不使用 MACD 检测背离
  --no-hist: 不使用 Hist（MACD 柱状图）检测背离
  --no-obv: 不使用 OBV 检测背离

输出文件：
  Excel 文件包含以下列：
  - 股票名称、股票代码
  - 是否通过（bool）
  - 当前 bar 的 AMP 指标（amp_strength, mid_slope, close_pos, channel_width, period）
  - 历史冲顶信息（latest_trigger_bar, peak_amp_strength, peak_close_pos, peak_mid_slope）
  - 背离信号（div_detected, div_type, div_bar_idx, div_indicator, div_distance）

输出示例：
  总股票数：500
  通过 AMP 条件群：15
  通过背离条件群：180
  通过全部条件：3
  
  通过全部条件的股票：
    恒逸石化 (000703.SZ) amp=0.950 slope=0.010 close_pos=0.485 div=hist(Positive Hidden)
    航天发展 (000547.SZ) amp=0.957 slope=0.024 close_pos=0.000 div=macd(Positive Regular)
    航天环宇 (688523.SH) amp=0.946 slope=0.019 close_pos=0.211 div=obv(Positive Hidden)

依赖：
  - pandas, numpy, tqdm
  - features/amp_plotly.py（AMP 核心计算）
  - features/divergence_many_plotly.py（背离检测）
  - datasource/pytdx_client.py（数据源）

注意事项：
  1) 首次运行需要连接 pytdx 数据源，确保网络畅通
  2) 默认扫描 stock_concepts_cache.xlsx 中的全部股票（约 5000 只）
  3) 使用 --max-stocks 参数可以先测试少量股票
  4) 输出文件会自动创建 output 目录
  5) mid_slope 使用 adjusted-space slope（对数空间），与线性空间口径不同

版本信息：
  - 核心计算逻辑来自 features/amp_plotly.py
  - slope 统一使用 adjusted-space slope，避免线性模式下因起点价格过小导致数值爆炸
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    # 关键：先注册到 sys.modules，避免 dataclass 在动态导入时拿不到 module namespace
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _candidate_module_paths(filename: str) -> List[str]:
    candidates = [
        os.path.join(THIS_DIR, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "features", filename),
        os.path.join(os.path.dirname(THIS_DIR), "features", filename),
        os.path.join(os.path.dirname(BASE_DIR), "features", filename),
    ]
    out: List[str] = []
    for p in candidates:
        if p not in out:
            out.append(p)
    return out


def _load_first_existing(module_name: str, filename: str):
    checked = _candidate_module_paths(filename)
    for p in checked:
        if os.path.exists(p):
            return _load_module_from_path(module_name, p)
    raise FileNotFoundError(f"未找到 {filename}，已检查路径: {checked}")


# AMP 模块：严格按 features/amp_plotly.py 当前版本对齐
try:
    from features.amp_plotly import AMPConfig, compute_amp_last
except Exception:
    amp_mod = _load_first_existing("amp_plotly_local", "amp_plotly.py")
    AMPConfig = amp_mod.AMPConfig
    compute_amp_last = amp_mod.compute_amp_last

# 背离模块
try:
    from features.divergence_many_plotly import (
        DivConfig,
        compute_indicators,
        pivots_confirmed,
        calculate_divs,
    )
except Exception:
    div_mod = _load_first_existing("divergence_many_plotly_local", "divergence_many_plotly.py")
    DivConfig = div_mod.DivConfig
    compute_indicators = div_mod.compute_indicators
    pivots_confirmed = div_mod.pivots_confirmed
    calculate_divs = div_mod.calculate_divs

from datasource.pytdx_client import connect_pytdx, get_kline_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_pytdx_api = None


def get_pytdx_api():
    global _pytdx_api
    if _pytdx_api is None:
        _pytdx_api = connect_pytdx()
    return _pytdx_api


def reconnect_pytdx():
    global _pytdx_api
    if _pytdx_api is not None:
        try:
            _pytdx_api.disconnect()
        except Exception:
            pass
    _pytdx_api = connect_pytdx()
    return _pytdx_api


def get_stock_pool() -> Dict[str, str]:
    xlsx_path = os.path.join(BASE_DIR, "stock_concepts_cache.xlsx")
    if not os.path.exists(xlsx_path):
        raise RuntimeError(f"股票池文件不存在：{xlsx_path}")

    df = pd.read_excel(xlsx_path)
    if "ts_code" not in df.columns or "name" not in df.columns:
        raise RuntimeError("股票池文件缺少必要列 (ts_code, name)")

    stock_dict = dict(zip(df["name"], df["ts_code"]))
    logger.info(f"从股票池读取 {len(stock_dict)} 只股票")
    return stock_dict


def get_k_data(ts_code: str, api, period: str = "d", count: int = 810, max_retries: int = 3) -> Optional[pd.DataFrame]:
    global _pytdx_api
    for attempt in range(max_retries):
        try:
            symbol = ts_code.split(".")[0]
            df = get_kline_data(api, symbol, period, count)
            if df is None or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return None
            df = df.rename(columns={"datetime": "bar_time"}).copy()
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
            if df.empty:
                return None
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"获取 {ts_code} 失败，尝试重连后重试：{e}")
                time.sleep(1.0)
                try:
                    api = reconnect_pytdx()
                except Exception:
                    pass
                continue
            logger.error(f"获取 {ts_code} 的 K 线数据失败：{e}")
            return None
    return None


# =========================
# AMP helpers (与 features/amp_plotly.py 同口径)
# =========================

def compute_amp_metrics_for_bar(df_hist: pd.DataFrame, cfg: AMPConfig, end_idx: int) -> Optional[Dict]:
    if end_idx < 59:
        return None

    df_view = df_hist.iloc[: end_idx + 1].copy()
    if len(df_view) < 60:
        return None

    try:
        payload = compute_amp_last(df_view, cfg)
    except Exception:
        return None

    m = payload["metrics"]
    return {
        "amp_strength": float(m["strength_pR_cD"]),
        "upper_slope": float(m["upper_slope"]),
        "lower_slope": float(m["lower_slope"]),
        "mid_slope": float(m["mid_slope"]),
        "close_pos": float(m["close_pos_0_1"]),
        "channel_width": float(m["channel_width_t"]),
        "period": int(m["finalPeriod"]),
        "window_len": int(m["window_len"]),
    }


def check_current_amp_quick_filter(current_metrics: Optional[Dict], cfg_args) -> bool:
    if current_metrics is None:
        return False
    mid_slope = current_metrics["mid_slope"]
    return (
        current_metrics["amp_strength"] >= cfg_args.amp_min_strength
        and np.isfinite(mid_slope)
        and mid_slope > cfg_args.amp_min_mid_slope
        and current_metrics["close_pos"] <= cfg_args.amp_max_close_pos
    )


def find_latest_peak_and_check_pullback_state(
    df: pd.DataFrame,
    target_idx: int,
    amp_cfg: AMPConfig,
    args,
) -> Tuple[bool, Optional[Dict], List[Dict]]:
    """
    查找最近一次强趋势冲顶，并检查当前是否处于回踩位置
    
    条件群 C 逻辑：
    - 只检查当前 bar 的 close_pos 是否 <= 0.50
    - 不关心中间过程
    """
    if target_idx < 60:
        return False, None, []

    start_idx = max(60, target_idx - args.trend_lookback_bars)
    cache: List[Dict] = []
    latest_peak: Optional[Dict] = None

    # 查找最近一次强趋势冲顶
    for i in range(start_idx, target_idx):
        mm = compute_amp_metrics_for_bar(df, amp_cfg, i)
        if mm is None:
            continue
        rec = {"bar_idx": i, **mm}
        cache.append(rec)
        if (
            mm["amp_strength"] >= args.peak_min_strength
            and mm["close_pos"] >= args.peak_min_close_pos
            and mm["mid_slope"] > args.peak_min_mid_slope
        ):
            latest_peak = rec

    if latest_peak is None:
        return False, None, cache

    # 只检查当前 bar 是否处于回踩位置 (close_pos <= 0.50)
    current_metrics = compute_amp_metrics_for_bar(df, amp_cfg, target_idx)
    if current_metrics is None:
        return False, latest_peak, cache
    
    pullback_position = current_metrics["close_pos"] <= args.amp_max_close_pos
    return pullback_position, latest_peak, cache


# =========================
# Divergence helpers
# =========================

def check_recent_bottom_divergence(
    df: pd.DataFrame,
    target_idx: int,
    cfg: DivConfig,
    lookback_days: int = 3,
) -> Tuple[bool, Optional[Dict]]:
    if target_idx < max(lookback_days, cfg.prd + 5):
        return False, None

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ind = compute_indicators(df)

    pivot_src_h = (df["close"] if cfg.source == "Close" else df["high"]).to_numpy(dtype=float)
    pivot_src_l = (df["close"] if cfg.source == "Close" else df["low"]).to_numpy(dtype=float)
    ph_conf, _ = pivots_confirmed(pivot_src_h, cfg.prd)
    _, pl_conf = pivots_confirmed(pivot_src_l, cfg.prd)

    maxarraysize = 20
    ph_positions: List[int] = []
    pl_positions: List[int] = []
    ph_vals: List[float] = []
    pl_vals: List[float] = []

    series_list = [
        ("macd", cfg.calcmacd),
        ("hist", cfg.calcmacda),
        ("obv", cfg.calcobv),
    ]

    bottom_divs: List[Dict] = []

    for t in range(min(target_idx + 1, len(df))):
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

        for key, enabled in series_list:
            if not enabled:
                continue

            src_arr = ind[key].to_numpy(dtype=float)
            divs4 = calculate_divs(
                enabled=True,
                src=src_arr,
                close=close,
                high=high,
                low=low,
                ph_positions=ph_positions.copy(),
                ph_vals=ph_vals.copy(),
                pl_positions=pl_positions.copy(),
                pl_vals=pl_vals.copy(),
                t=t,
                cfg=cfg,
            )

            # 只接受底背离：0=Positive Regular, 2=Positive Hidden
            for div_type in (0, 2):
                dist = divs4[div_type]
                if dist > 0 and (target_idx - lookback_days + 1) <= t <= target_idx:
                    bottom_divs.append(
                        {
                            "bar_idx": t,
                            "indicator": key,
                            "divergence_type": "Positive Regular" if div_type == 0 else "Positive Hidden",
                            "distance": float(dist),
                        }
                    )

    if not bottom_divs:
        return False, None

    latest = bottom_divs[-1]
    div_age = target_idx - latest["bar_idx"]
    return True, {
        "bar_idx": latest["bar_idx"],
        "indicator": latest["indicator"],
        "divergence_type": latest["divergence_type"],
        "distance": latest["distance"],
        "age": div_age,
        "total_count": len(bottom_divs),
    }


# =========================
# Main picker
# =========================

def run_stock_picker(target_date: str, max_stocks: Optional[int], args) -> pd.DataFrame:
    logger.info(f"开始运行选股策略，目标日期：{target_date}")

    api = get_pytdx_api()
    stock_pool = get_stock_pool()
    if max_stocks:
        stock_pool = dict(list(stock_pool.items())[:max_stocks])
        logger.info(f"限制处理前 {max_stocks} 只股票")

    amp_cfg = AMPConfig()
    div_cfg = DivConfig(only_pos_divs=True)
    results = []

    for stock_name, ts_code in tqdm(stock_pool.items(), desc="选股进度", unit="只"):
        df = get_k_data(ts_code, api, period="d", count=args.bars)
        if df is None or len(df) < args.min_bars_required:
            continue

        mask = df["bar_time"].astype(str).str.startswith(target_date)
        if not mask.any():
            continue
        target_idx = int(df[mask].index[0])

        current_metrics = compute_amp_metrics_for_bar(df, amp_cfg, target_idx)
        amp_current_pass = check_current_amp_quick_filter(current_metrics, args)
        if not amp_current_pass:
            results.append(
                {
                    "股票名称": stock_name,
                    "股票代码": ts_code,
                    "AMP当前bar快速过滤通过": False,
                    "最近3天底背离通过": False,
                    "历史趋势/第一次回踩通过": False,
                    "全部通过": False,
                    "amp_strength": current_metrics["amp_strength"] if current_metrics else None,
                    "mid_slope": current_metrics["mid_slope"] if current_metrics else None,
                    "upper_slope": current_metrics["upper_slope"] if current_metrics else None,
                    "lower_slope": current_metrics["lower_slope"] if current_metrics else None,
                    "close_pos": current_metrics["close_pos"] if current_metrics else None,
                    "channel_width": current_metrics["channel_width"] if current_metrics else None,
                    "period": current_metrics["period"] if current_metrics else None,
                    "window_len": current_metrics["window_len"] if current_metrics else None,
                    "history_peak_bar_idx": None,
                    "history_peak_amp_strength": None,
                    "history_peak_mid_slope": None,
                    "history_peak_close_pos": None,
                    "divergence_bar_idx": None,
                    "divergence_indicator": None,
                    "divergence_type": None,
                    "divergence_distance": None,
                    "divergence_count": None,
                }
            )
            continue

        div_passed, div_info = check_recent_bottom_divergence(df, target_idx, div_cfg, lookback_days=args.div_lookback_days)
        if not div_passed:
            results.append(
                {
                    "股票名称": stock_name,
                    "股票代码": ts_code,
                    "AMP当前bar快速过滤通过": True,
                    "最近3天底背离通过": False,
                    "历史趋势/第一次回踩通过": False,
                    "全部通过": False,
                    "amp_strength": current_metrics["amp_strength"],
                    "mid_slope": current_metrics["mid_slope"],
                    "upper_slope": current_metrics["upper_slope"],
                    "lower_slope": current_metrics["lower_slope"],
                    "close_pos": current_metrics["close_pos"],
                    "channel_width": current_metrics["channel_width"],
                    "period": current_metrics["period"],
                    "window_len": current_metrics["window_len"],
                    "history_peak_bar_idx": None,
                    "history_peak_amp_strength": None,
                    "history_peak_mid_slope": None,
                    "history_peak_close_pos": None,
                    "divergence_bar_idx": None,
                    "divergence_indicator": None,
                    "divergence_type": None,
                    "divergence_distance": None,
                    "divergence_count": None,
                }
            )
            continue

        hist_passed, latest_peak, _ = find_latest_peak_and_check_pullback_state(df, target_idx, amp_cfg, args)
        results.append(
            {
                "股票名称": stock_name,
                "股票代码": ts_code,
                "AMP当前bar快速过滤通过": True,
                "最近3天底背离通过": True,
                "历史趋势/第一次回踩通过": hist_passed,
                "全部通过": bool(hist_passed),
                "amp_strength": current_metrics["amp_strength"],
                "mid_slope": current_metrics["mid_slope"],
                "upper_slope": current_metrics["upper_slope"],
                "lower_slope": current_metrics["lower_slope"],
                "close_pos": current_metrics["close_pos"],
                "channel_width": current_metrics["channel_width"],
                "period": current_metrics["period"],
                "window_len": current_metrics["window_len"],
                "history_peak_bar_idx": latest_peak["bar_idx"] if latest_peak else None,
                "history_peak_amp_strength": latest_peak["amp_strength"] if latest_peak else None,
                "history_peak_mid_slope": latest_peak["mid_slope"] if latest_peak else None,
                "history_peak_close_pos": latest_peak["close_pos"] if latest_peak else None,
                "divergence_bar_idx": div_info["bar_idx"] if div_info else None,
                "divergence_indicator": div_info["indicator"] if div_info else None,
                "divergence_type": div_info["divergence_type"] if div_info else None,
                "divergence_distance": div_info["distance"] if div_info else None,
                "divergence_age": div_info["age"] if div_info else None,
                "divergence_count": div_info["total_count"] if div_info else None,
            }
        )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    logger.info("\n选股结果统计:")
    logger.info(f"  总股票数：{len(result_df)}")
    logger.info(f"  AMP当前bar快速过滤通过：{int(result_df['AMP当前bar快速过滤通过'].sum())}")
    logger.info(f"  最近3天底背离通过：{int(result_df['最近3天底背离通过'].sum())}")
    logger.info(f"  全部通过：{int(result_df['全部通过'].sum())}")
    return result_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="选股策略脚本（AMP + 底背离）")
    p.add_argument("--date", type=str, required=True, help="目标日期 YYYY-MM-DD")
    p.add_argument("--max-stocks", type=int, default=None, help="最大处理股票数（测试用）")
    p.add_argument("--output", type=str, default=None, help="输出 Excel 路径")
    p.add_argument("--bars", type=int, default=810, help="拉取日线 bars 数")
    p.add_argument("--min-bars-required", type=int, default=220, help="最少样本 bars")

    p.add_argument("--amp-min-strength", type=float, default=0.70, help="当前/回踩 AMP 最小强度")
    p.add_argument("--amp-min-mid-slope", type=float, default=0.001, help="当前/回踩最小 mid_slope（与 features/amp_plotly.py 同口径）")
    p.add_argument("--amp-max-close-pos", type=float, default=0.50, help="当前/回踩最大 close_pos")

    p.add_argument("--trend-lookback-bars", type=int, default=100, help="历史扫描回看 bars")
    p.add_argument("--peak-min-strength", type=float, default=0.70, help="峰值最小强度")
    p.add_argument("--peak-min-close-pos", type=float, default=0.98, help="峰值最小 close_pos")
    p.add_argument("--peak-min-mid-slope", type=float, default=0.001, help="峰值最小 mid_slope")

    p.add_argument("--div-lookback-days", type=int, default=3, help="底背离回看天数")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result_df = run_stock_picker(args.date, args.max_stocks, args)
    if result_df.empty:
        logger.info("没有结果")
        return

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(os.getcwd(), f"selected_stocks_{args.date}.xlsx")

    result_df.to_excel(out_path, index=False)
    logger.info(f"结果已保存：{out_path}")


if __name__ == "__main__":
    main()
