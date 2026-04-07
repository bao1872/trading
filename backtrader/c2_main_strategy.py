# -*- coding: utf-8 -*-
"""
C2 主版策略脚本（日线低位 + 转强确认 + 严格周线 DSA 位置过滤）

主版规则
--------
周线过滤：
    w_dsa_pivot_pos_01 < 0.7

日线条件：
    dsa_pivot_pos_01 <= 0.30
    signed_vwap_dev_pct <= -1.0  # VWAP 偏离度阈值
    bars_since_dir_change <= 3

交易口径
--------
- 信号在当日收盘确认，并按当日收盘买入
- 固定持有 40 个 bar，于到期当日收盘卖出（无止损）
- 交易成本：
    * 佣金：单边 0.1%
    * 印花税：卖出 0.05%
    * 滑点：双边 0.2%（买入 0.1%，卖出 0.1%）

用法
----
1) 单股回测：
   python backtrader/c2_main_strategy.py --symbol 600547 --bars 1200

2) 选股模式（指定日期扫描全市场，结果保存到数据库）：
   python backtrader/c2_main_strategy.py --select-date 2024-01-15

3) 选股模式（限制扫描数量，用于测试）：
   python backtrader/c2_main_strategy.py --select-date 2024-01-15 --max-stocks 100

4) 回补模式（计算历史日期范围内的选股结果）：
   python backtrader/c2_main_strategy.py --backfill 2025-03-01

5) 回补模式（指定日期范围）：
   python backtrader/c2_main_strategy.py --backfill 2024-01-01 --end-date 2024-03-01

6) 回补模式（限制扫描数量，用于测试）：
   python backtrader/c2_main_strategy.py --backfill 2024-01-01 --end-date 2024-01-10 --max-stocks 100

输出
----
1) 交易明细 CSV（单股回测模式）
2) 策略汇总 CSV（单股回测模式）
3) 可选 HTML 图（单股回测模式）
4) 选股结果保存到数据库 c2_strategy_selections 表（选股/回补模式）
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 因子脚本路径（从脚本头部硬编码引入）
DEFAULT_FACTORS_SCRIPT = (
    Path(__file__).resolve().parent.parent / "features" / "merged_dsa_atr_rope_bb_factors.py"
)


@dataclass
class StrategyConfig:
    dsa_thr: float = 0.30
    vwap_thr: float = -1.0
    bars_thr: int = 3
    weekly_pos_thr: float = 0.70
    require_break_up: bool = True
    hold_bars: int = 40
    cooldown_bars: int = 20
    weekly_resample_rule: str = "W-FRI"
    commission_rate: float = 0.001
    stamp_tax_rate: float = 0.0005
    slippage_side: float = 0.001


class FactorEngine:
    def __init__(self):
        script = str(DEFAULT_FACTORS_SCRIPT)
        if not Path(script).exists():
            raise FileNotFoundError(f"找不到因子脚本: {script}")
        self.module = _load_module_from_path("merged_factors_module_c2_same_day", script)
        self.fetch_kline_pytdx = self.module.fetch_kline_pytdx
        self.compute_dsa = self.module.compute_dsa
        self.compute_atr_rope = self.module.compute_atr_rope
        self.compute_bollinger = self.module.compute_bollinger
        self.DSAConfig = self.module.DSAConfig
        self.RopeConfig = self.module.RopeConfig

    def fetch_daily(self, symbol: str, bars: int) -> pd.DataFrame:
        df = self.fetch_kline_pytdx(symbol=symbol, freq="d", count=bars)
        if df is None or df.empty:
            raise RuntimeError(f"{symbol} 无法获取日线数据")
        if "volume" not in df.columns:
            raise RuntimeError("日线数据缺少 volume 列")
        return df.copy()

    def compute_daily_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        dsa_df, _, _ = self.compute_dsa(df, self.DSAConfig(prd=50, base_apt=20.0))
        rope_df = self.compute_atr_rope(df, self.RopeConfig(length=14, multi=1.5, source="close"))
        bb_df = self.compute_bollinger(df, length=20, mult=2.0, pct_lookback=120)
        return pd.concat(
            [df, dsa_df, rope_df.drop(columns=df.columns, errors="ignore"), bb_df.drop(columns=df.columns, errors="ignore")],
            axis=1,
        )

    def compute_weekly_strict_dsa(self, daily_df: pd.DataFrame, weekly_rule: str = "W-FRI") -> pd.DataFrame:
        weekly = pd.DataFrame(index=daily_df.resample(weekly_rule, label="right", closed="right").last().index)
        weekly["open"] = daily_df["open"].resample(weekly_rule, label="right", closed="right").first()
        weekly["high"] = daily_df["high"].resample(weekly_rule, label="right", closed="right").max()
        weekly["low"] = daily_df["low"].resample(weekly_rule, label="right", closed="right").min()
        weekly["close"] = daily_df["close"].resample(weekly_rule, label="right", closed="right").last()
        weekly["volume"] = daily_df["volume"].resample(weekly_rule, label="right", closed="right").sum(min_count=1)
        weekly = weekly.dropna(subset=["open", "high", "low", "close"])
        out = pd.DataFrame(index=daily_df.index)
        out["w_DSA_DIR"] = np.nan
        out["w_dsa_pivot_pos_01"] = np.nan
        out["w_signed_vwap_dev_pct"] = np.nan
        if len(weekly) < 20:
            return out
        w_dsa, _, _ = self.compute_dsa(weekly, self.DSAConfig(prd=50, base_apt=20.0))
        w_keep = w_dsa[["DSA_DIR", "dsa_pivot_pos_01", "signed_vwap_dev_pct"]].rename(
            columns={
                "DSA_DIR": "w_DSA_DIR",
                "dsa_pivot_pos_01": "w_dsa_pivot_pos_01",
                "signed_vwap_dev_pct": "w_signed_vwap_dev_pct",
            }
        )
        return w_keep.shift(1).reindex(daily_df.index, method="ffill")


def build_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    required = ["w_dsa_pivot_pos_01", "dsa_pivot_pos_01", "signed_vwap_dev_pct", "bars_since_dir_change", "rope_dir"]
    if cfg.require_break_up:
        required.append("range_break_up")
    for col in required:
        if col not in df.columns:
            raise KeyError(f"缺少必要字段: {col}")
    sig = (
        (df["w_dsa_pivot_pos_01"] < cfg.weekly_pos_thr)
        & (df["dsa_pivot_pos_01"] <= cfg.dsa_thr)
        & (df["signed_vwap_dev_pct"] <= cfg.vwap_thr)
        & (df["rope_dir"] == 1)
        & (df["bars_since_dir_change"] <= cfg.bars_thr)
    )
    if cfg.require_break_up:
        sig &= (df["range_break_up"] == 1)
    return sig.fillna(False)


def dedup_signals(signal_idx: List[int], cooldown_bars: int) -> List[int]:
    kept: List[int] = []
    last = -10**9
    for idx in signal_idx:
        if idx - last >= cooldown_bars:
            kept.append(idx)
            last = idx
    return kept


def prepare_stop_reference(df: pd.DataFrame) -> pd.Series:
    if "rope_dir" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="stop_ref_price")
    flip_to_up = (df["rope_dir"] == 1) & (df["rope_dir"].shift(1).fillna(0) != 1)
    stop_ref = df["low"].shift(1).where(flip_to_up).ffill()
    stop_ref.name = "stop_ref_price"
    return stop_ref


def apply_entry_cost(raw_price: float, cfg: StrategyConfig) -> float:
    return raw_price * (1.0 + cfg.commission_rate + cfg.slippage_side)


def apply_exit_cost(raw_price: float, cfg: StrategyConfig) -> float:
    return raw_price * (1.0 - cfg.commission_rate - cfg.stamp_tax_rate - cfg.slippage_side)


def generate_trades(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    work = df.copy()
    signal = build_signals(work, cfg)
    signal_indices = dedup_signals(np.flatnonzero(signal.to_numpy()).tolist(), cfg.cooldown_bars)

    close_arr = work["close"].to_numpy(float)
    low_arr = work["low"].to_numpy(float)
    rows: List[Dict] = []

    for idx in signal_indices:
        if idx + cfg.hold_bars >= len(work):
            continue

        raw_entry_price = close_arr[idx]
        if not np.isfinite(raw_entry_price) or raw_entry_price <= 0:
            continue

        entry_price = apply_entry_cost(raw_entry_price, cfg)
        exit_idx = idx + cfg.hold_bars
        raw_exit_price = close_arr[exit_idx]
        exit_reason = "hold_40"

        # 无止损，固定持有cfg.hold_bars后退出

        exit_price = apply_exit_cost(raw_exit_price, cfg)
        ret = (exit_price - entry_price) / entry_price if entry_price > 0 else np.nan

        if exit_idx > idx:
            future_low = np.nanmin(low_arr[idx + 1: exit_idx + 1])
            mae = (future_low - entry_price) / entry_price if entry_price > 0 else np.nan
        else:
            mae = np.nan

        rr = ret / abs(mae) if pd.notna(ret) and pd.notna(mae) and mae != 0 else np.nan

        rows.append(
            {
                "symbol": work["symbol"].iloc[idx] if "symbol" in work.columns else "",
                "entry_idx": idx,
                "entry_time": work.index[idx],
                "entry_price_raw": raw_entry_price,
                "entry_price": entry_price,
                "stop_price": np.nan,  # 无止损
                "exit_idx": exit_idx,
                "exit_time": work.index[exit_idx],
                "exit_price_raw": raw_exit_price,
                "exit_price": exit_price,
                "hold_bars": exit_idx - idx,
                "planned_hold_bars": cfg.hold_bars,
                "exit_reason": exit_reason,
                "ret": ret,
                "mae": mae,
                "rr": rr,
                "win": float(ret > 0) if pd.notna(ret) else np.nan,
                "w_dsa_pivot_pos_01": work["w_dsa_pivot_pos_01"].iloc[idx],
                "w_DSA_DIR": work["w_DSA_DIR"].iloc[idx] if "w_DSA_DIR" in work.columns else np.nan,
                "w_signed_vwap_dev_pct": work["w_signed_vwap_dev_pct"].iloc[idx] if "w_signed_vwap_dev_pct" in work.columns else np.nan,
                "dsa_pivot_pos_01": work["dsa_pivot_pos_01"].iloc[idx],
                "signed_vwap_dev_pct": work["signed_vwap_dev_pct"].iloc[idx],
                "bars_since_dir_change": work["bars_since_dir_change"].iloc[idx],
                "rope_dir": work["rope_dir"].iloc[idx] if "rope_dir" in work.columns else np.nan,
                "rope_slope_atr_5": work["rope_slope_atr_5"].iloc[idx] if "rope_slope_atr_5" in work.columns else np.nan,
                "range_break_up": work["range_break_up"].iloc[idx] if "range_break_up" in work.columns else np.nan,
                "range_break_up_strength": work["range_break_up_strength"].iloc[idx] if "range_break_up_strength" in work.columns else np.nan,
                "bb_pos_01": work["bb_pos_01"].iloc[idx] if "bb_pos_01" in work.columns else np.nan,
                "bb_width_percentile": work["bb_width_percentile"].iloc[idx] if "bb_width_percentile" in work.columns else np.nan,
            }
        )

    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{
            "symbol": symbol, "trade_n": 0, "ret_mean": np.nan, "ret_median": np.nan,
            "mae_mean": np.nan, "wr": np.nan, "rr_mean": np.nan,
            "strategy_dsa_thr": cfg.dsa_thr, "strategy_vwap_thr": cfg.vwap_thr,
            "strategy_bars_thr": cfg.bars_thr, "strategy_weekly_pos_thr": cfg.weekly_pos_thr,
            "strategy_break_up_required": int(cfg.require_break_up),
            "strategy_hold_bars": cfg.hold_bars, "strategy_cooldown_bars": cfg.cooldown_bars,
            "commission_rate": cfg.commission_rate, "stamp_tax_rate": cfg.stamp_tax_rate,
            "slippage_side": cfg.slippage_side,
        }])

    return pd.DataFrame([{
        "symbol": symbol,
        "trade_n": int(len(trades)),
        "ret_mean": float(trades["ret"].mean()),
        "ret_median": float(trades["ret"].median()),
        "mae_mean": float(trades["mae"].mean()),
        "wr": float(trades["win"].mean()),
        "rr_mean": float(trades["rr"].mean()),
        "strategy_dsa_thr": cfg.dsa_thr,
        "strategy_vwap_thr": cfg.vwap_thr,
        "strategy_bars_thr": cfg.bars_thr,
        "strategy_weekly_pos_thr": cfg.weekly_pos_thr,
        "strategy_break_up_required": int(cfg.require_break_up),
        "strategy_hold_bars": cfg.hold_bars,
        "strategy_cooldown_bars": cfg.cooldown_bars,
        "commission_rate": cfg.commission_rate,
        "stamp_tax_rate": cfg.stamp_tax_rate,
        "slippage_side": cfg.slippage_side,
    }])


def build_html(df: pd.DataFrame, trades: pd.DataFrame, out_html: str, symbol: str) -> None:
    x = np.arange(len(df), dtype=float)
    tick_text = [dt.strftime("%Y-%m-%d") for dt in df.index]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=(f"{symbol} C2 主版策略", "DSA / Rope 关键信号", "周线 DSA 位置 / 日线 DSA 位置"),
    )
    fig.add_trace(go.Candlestick(x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 showlegend=False, name="K线"), row=1, col=1)
    if "DSA_VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["DSA_VWAP"], mode="lines", name="DSA_VWAP", line=dict(width=2)), row=1, col=1)
    if "rope" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["rope"], mode="lines", name="rope", line=dict(width=2)), row=1, col=1)
    for _, tr in trades.iterrows():
        ei, xi = int(tr["entry_idx"]), int(tr["exit_idx"])
        fig.add_trace(go.Scatter(x=[ei], y=[df["close"].iloc[ei]], mode="markers",
                                 marker=dict(size=10, symbol="triangle-up", color="green"),
                                 name="买点", showlegend=False,
                                 hovertemplate=f"买入 {tr['entry_time']}<br>价格={tr['entry_price']:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xi], y=[df["close"].iloc[xi]], mode="markers",
                                 marker=dict(size=10, symbol="triangle-down", color="red"),
                                 name="卖点", showlegend=False,
                                 hovertemplate=f"卖出 {tr['exit_time']}<br>价格={tr['exit_price']:.3f}<br>{tr['exit_reason']}<extra></extra>"), row=1, col=1)

    if "signed_vwap_dev_pct" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["signed_vwap_dev_pct"], mode="lines", name="signed_vwap_dev_pct"), row=2, col=1)
    if "bars_since_dir_change" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["bars_since_dir_change"], mode="lines", name="bars_since_dir_change"), row=2, col=1)
    if "range_break_up" in df.columns:
        up_mask = df["range_break_up"] == 1
        fig.add_trace(go.Scatter(
            x=x[up_mask.to_numpy()],
            y=df.loc[up_mask, "close"],
            mode="markers",
            marker=dict(size=8, symbol="circle-open", color="blue"),
            name="break_up",
            showlegend=False,
        ), row=1, col=1)
    if "w_dsa_pivot_pos_01" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["w_dsa_pivot_pos_01"], mode="lines", name="w_dsa_pivot_pos_01"), row=3, col=1)
    if "dsa_pivot_pos_01" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["dsa_pivot_pos_01"], mode="lines", name="dsa_pivot_pos_01"), row=3, col=1)

    step = max(1, len(x) // 12)
    fig.update_xaxes(tickmode="array", tickvals=x[::step], ticktext=tick_text[::step])
    fig.update_layout(height=950, title=f"{symbol} C2 主版策略", hovermode="x unified")
    fig.write_html(out_html)


def get_stock_pool_from_db() -> List[str]:
    """从数据库获取股票池列表"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasource.database import get_engine
    engine = get_engine()
    query = "SELECT ts_code as code FROM stock_pools ORDER BY ts_code"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df['code'].tolist()


def load_daily_from_db(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从数据库加载日线数据"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasource.k_data_loader import load_k_data
    df = load_k_data(ts_code=ts_code, freq='d', start_date=start_date, end_date=end_date)
    if df.empty:
        return df
    # 重命名列以兼容因子计算
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    })
    return df


def check_signal_on_date(df: pd.DataFrame, cfg: StrategyConfig, target_date: pd.Timestamp) -> Optional[Dict]:
    """
    检查指定日期是否满足买入信号

    Returns:
        如果满足信号，返回包含指标的字典；否则返回 None
    """
    # 将索引转换为日期（去除时间部分）进行匹配
    df_dates = df.index.normalize()
    target_date_normalized = target_date.normalize() if hasattr(target_date, 'normalize') else pd.Timestamp(target_date).normalize()

    if target_date_normalized not in df_dates:
        return None

    # 获取匹配的行
    row = df.loc[df_dates == target_date_normalized].iloc[0]

    # 检查必要字段是否存在
    required = ["w_dsa_pivot_pos_01", "dsa_pivot_pos_01", "signed_vwap_dev_pct",
                "bars_since_dir_change", "rope_dir", "close"]
    if cfg.require_break_up:
        required.append("range_break_up")
    for col in required:
        if col not in row.index or pd.isna(row[col]):
            return None

    # 检查信号条件
    if not (row["w_dsa_pivot_pos_01"] < cfg.weekly_pos_thr):
        return None
    if not (row["dsa_pivot_pos_01"] <= cfg.dsa_thr):
        return None
    if not (row["signed_vwap_dev_pct"] <= cfg.vwap_thr):
        return None
    if not (row["rope_dir"] == 1):
        return None
    if not (row["bars_since_dir_change"] <= cfg.bars_thr):
        return None
    if cfg.require_break_up and not (row["range_break_up"] == 1):
        return None

    # 满足信号，返回指标
    result = {
        "signal_date": target_date.strftime("%Y-%m-%d"),
        "close": float(row["close"]),
        "dsa_pivot_pos_01": float(row["dsa_pivot_pos_01"]),
        "signed_vwap_dev_pct": float(row["signed_vwap_dev_pct"]),
        "w_dsa_pivot_pos_01": float(row["w_dsa_pivot_pos_01"]),
        "bars_since_dir_change": int(row["bars_since_dir_change"]),
        "rope_dir": int(row["rope_dir"]),
        "rope_slope_atr_5": float(row["rope_slope_atr_5"]) if "rope_slope_atr_5" in row.index and pd.notna(row["rope_slope_atr_5"]) else np.nan,
        "range_break_up": int(row["range_break_up"]) if "range_break_up" in row.index and pd.notna(row["range_break_up"]) else 0,
        "range_break_up_strength": float(row["range_break_up_strength"]) if "range_break_up_strength" in row.index and pd.notna(row["range_break_up_strength"]) else np.nan,
        "bb_pos_01": float(row["bb_pos_01"]) if "bb_pos_01" in row.index and pd.notna(row["bb_pos_01"]) else np.nan,
        "bb_width_percentile": float(row["bb_width_percentile"]) if "bb_width_percentile" in row.index and pd.notna(row["bb_width_percentile"]) else np.nan,
    }
    return result


def scan_stocks_for_signals(select_date: str, cfg: StrategyConfig, max_stocks: Optional[int] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    扫描股票池，找出指定日期满足买入信号的股票（从数据库读取数据）

    Args:
        select_date: 选股日期，格式 "YYYY-MM-DD"
        cfg: 策略配置
        max_stocks: 最大扫描股票数量（None 表示全部）
        symbols: 自定义股票列表（None 表示从数据库获取全部）

    Returns:
        DataFrame 包含所有满足条件的股票及其指标
    """
    engine = FactorEngine()
    target_date = pd.Timestamp(select_date)

    # 计算需要的数据范围（约 2 年，足够计算周线 DSA）
    start_date = (target_date - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = (target_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # 获取股票池
    if symbols is None:
        print("正在从数据库获取股票池...")
        symbols = get_stock_pool_from_db()
    if max_stocks is not None and max_stocks > 0:
        symbols = symbols[:max_stocks]
    print(f"股票池共 {len(symbols)} 只股票")
    print(f"数据范围: {start_date} ~ {end_date}")

    results = []
    errors = []

    for symbol in tqdm(symbols, desc="扫描股票", ncols=80, position=0, leave=True):
        try:
            # 从数据库加载日线数据
            raw = load_daily_from_db(symbol, start_date, end_date)
            if raw is None or raw.empty or len(raw) < 100:
                continue

            # 检查目标日期是否在数据范围内（使用normalize匹配日期）
            raw_dates = raw.index.normalize()
            target_date_normalized = target_date.normalize() if hasattr(target_date, 'normalize') else pd.Timestamp(target_date).normalize()
            if target_date_normalized not in raw_dates:
                continue

            raw["symbol"] = symbol
            daily = engine.compute_daily_factors(raw)
            weekly = engine.compute_weekly_strict_dsa(raw, cfg.weekly_resample_rule)
            df = pd.concat([daily, weekly], axis=1)

            # 检查指定日期是否有信号
            signal_data = check_signal_on_date(df, cfg, target_date)
            if signal_data:
                signal_data["symbol"] = symbol
                results.append(signal_data)

        except Exception as e:
            errors.append((symbol, str(e)))
            continue

    if errors:
        print(f"\n处理失败 {len(errors)} 只股票（已跳过）")

    if not results:
        print(f"\n在 {select_date} 未找到满足条件的股票")
        return pd.DataFrame()

    df_result = pd.DataFrame(results)
    # 调整列顺序
    cols = ["symbol", "signal_date", "close", "dsa_pivot_pos_01", "signed_vwap_dev_pct",
            "w_dsa_pivot_pos_01", "bars_since_dir_change", "rope_dir", "rope_slope_atr_5",
            "range_break_up", "range_break_up_strength", "bb_pos_01", "bb_width_percentile"]
    df_result = df_result[cols]

    return df_result


def save_selection_to_db(df: pd.DataFrame, select_date: str, cfg: StrategyConfig) -> int:
    """
    将选股结果保存到数据库

    Returns:
        保存的记录数
    """
    if df.empty:
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasource.database import get_engine

    engine = get_engine()

    # 准备数据
    df_db = df.copy()
    df_db['select_date'] = select_date
    df_db['created_at'] = pd.Timestamp.now()
    df_db['dsa_thr'] = cfg.dsa_thr
    df_db['vwap_thr'] = cfg.vwap_thr
    df_db['bars_thr'] = cfg.bars_thr
    df_db['weekly_pos_thr'] = cfg.weekly_pos_thr
    df_db['require_break_up'] = int(cfg.require_break_up)

    # 确保表存在
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS c2_strategy_selections (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        select_date DATE NOT NULL,
        signal_date DATE NOT NULL,
        close NUMERIC(12,4),
        dsa_pivot_pos_01 NUMERIC(10,6),
        signed_vwap_dev_pct NUMERIC(10,4),
        w_dsa_pivot_pos_01 NUMERIC(10,6),
        bars_since_dir_change INTEGER,
        rope_dir INTEGER,
        rope_slope_atr_5 NUMERIC(10,6),
        range_break_up INTEGER,
        range_break_up_strength NUMERIC(10,6),
        bb_pos_01 NUMERIC(10,6),
        bb_width_percentile NUMERIC(10,6),
        dsa_thr NUMERIC(10,4),
        vwap_thr NUMERIC(10,4),
        bars_thr INTEGER,
        weekly_pos_thr NUMERIC(10,4),
        require_break_up INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, select_date)
    );
    CREATE INDEX IF NOT EXISTS idx_c2_selection_date ON c2_strategy_selections(select_date);
    CREATE INDEX IF NOT EXISTS idx_c2_selection_symbol ON c2_strategy_selections(symbol);
    ALTER TABLE c2_strategy_selections ADD COLUMN IF NOT EXISTS range_break_up INTEGER;
    ALTER TABLE c2_strategy_selections ADD COLUMN IF NOT EXISTS range_break_up_strength NUMERIC(10,6);
    ALTER TABLE c2_strategy_selections ADD COLUMN IF NOT EXISTS require_break_up INTEGER DEFAULT 1;
    """

    with engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(text(create_table_sql))
        conn.commit()

    # 删除该日期的旧数据（避免重复）
    delete_sql = "DELETE FROM c2_strategy_selections WHERE select_date = :select_date"
    with engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(text(delete_sql), {'select_date': select_date})
        conn.commit()

    # 插入新数据
    df_db = df_db.rename(columns={'signal_date': 'signal_date'})
    columns = ['symbol', 'select_date', 'signal_date', 'close', 'dsa_pivot_pos_01',
               'signed_vwap_dev_pct', 'w_dsa_pivot_pos_01', 'bars_since_dir_change',
               'rope_dir', 'rope_slope_atr_5', 'range_break_up', 'range_break_up_strength', 'bb_pos_01', 'bb_width_percentile',
               'dsa_thr', 'vwap_thr', 'bars_thr', 'weekly_pos_thr', 'require_break_up', 'created_at']
    df_db = df_db[columns]

    df_db.to_sql('c2_strategy_selections', engine, if_exists='append', index=False)

    return len(df_db)


def get_trade_dates(start_date: str, end_date: str) -> List[str]:
    """
    获取指定范围内的交易日列表（从数据库的日线数据中推断）
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasource.database import get_engine
    from sqlalchemy import text

    engine = get_engine()

    # 从 stock_k_data 表中获取交易日列表
    query = """
    SELECT DISTINCT bar_time::date as trade_date
    FROM stock_k_data
    WHERE freq = 'd'
      AND bar_time::date BETWEEN :start_date AND :end_date
    ORDER BY trade_date
    """

    with engine.connect() as conn:
        result = conn.execute(text(query), {
            'start_date': start_date,
            'end_date': end_date
        })
        dates = [str(row[0]) for row in result]

    return dates


def backfill_selections(start_date: str, end_date: Optional[str], cfg: StrategyConfig, max_stocks: Optional[int] = None):
    """
    回补计算指定日期范围内的选股结果（优化版：一次性加载数据，避免重复查询）

    Args:
        start_date: 开始日期，格式 "YYYY-MM-DD"
        end_date: 结束日期，格式 "YYYY-MM-DD"（None 表示到今天）
        cfg: 策略配置
        max_stocks: 最大扫描股票数量（None 表示全部）
    """
    # 如果没有指定结束日期，使用今天
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"C2 策略回补模式")
    print(f"{'='*60}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"选股条件:")
    print(f"  - 周线 DSA 位置 < {cfg.weekly_pos_thr}")
    print(f"  - 日线 DSA 位置 <= {cfg.dsa_thr}")
    print(f"  - VWAP 偏离度 <= {cfg.vwap_thr}%")
    print(f"  - Rope 方向 = 1（上升）")
    print(f"  - 趋势转变后 bars <= {cfg.bars_thr}")
    print(f"{'='*60}\n")

    # 获取交易日列表
    trade_dates = get_trade_dates(start_date, end_date)
    print(f"共 {len(trade_dates)} 个交易日需要处理\n")

    # 计算统一的数据范围（覆盖整个回补期间）
    # 往前推 300 个交易日（约 1.5 年，足够计算周线 DSA）
    data_start = (pd.Timestamp(start_date) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    data_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # 获取股票池
    print("正在从数据库获取股票池...")
    symbols = get_stock_pool_from_db()
    if max_stocks is not None and max_stocks > 0:
        symbols = symbols[:max_stocks]
    print(f"股票池共 {len(symbols)} 只股票")
    print(f"统一数据范围: {data_start} ~ {data_end}\n")

    # 一次性加载所有股票的因子数据
    engine = FactorEngine()
    stock_data = {}

    for symbol in tqdm(symbols, desc="加载数据", ncols=80):
        try:
            raw = load_daily_from_db(symbol, data_start, data_end)
            if raw is None or raw.empty or len(raw) < 100:
                continue

            raw["symbol"] = symbol
            daily = engine.compute_daily_factors(raw)
            weekly = engine.compute_weekly_strict_dsa(raw, cfg.weekly_resample_rule)
            df = pd.concat([daily, weekly], axis=1)

            stock_data[symbol] = df
        except Exception:
            continue

    print(f"\n成功加载 {len(stock_data)} 只股票数据\n")

    # 逐个日期检查信号
    total_selected = 0
    dates_with_signals = 0

    for date in tqdm(trade_dates, desc="回补进度", ncols=80, position=0, leave=True):
        try:
            target_date = pd.Timestamp(date)

            results = []
            for symbol, df in stock_data.items():
                signal_data = check_signal_on_date(df, cfg, target_date)
                if signal_data:
                    signal_data["symbol"] = symbol
                    results.append(signal_data)

            if results:
                selection = pd.DataFrame(results)
                cols = ["symbol", "signal_date", "close", "dsa_pivot_pos_01", "signed_vwap_dev_pct",
                        "w_dsa_pivot_pos_01", "bars_since_dir_change", "rope_dir", "rope_slope_atr_5",
                        "range_break_up", "range_break_up_strength", "bb_pos_01", "bb_width_percentile"]
                selection = selection[cols]

                saved_count = save_selection_to_db(selection, date, cfg)
                total_selected += saved_count
                dates_with_signals += 1
                print(f"\n  {date}: 选出 {saved_count} 只股票")
        except Exception as e:
            print(f"\n  {date}: 处理失败 - {e}")
            continue

    print(f"\n{'='*60}")
    print(f"回补完成")
    print(f"{'='*60}")
    print(f"处理日期: {len(trade_dates)} 个")
    print(f"有信号的日期: {dates_with_signals} 个")
    print(f"总选股数: {total_selected} 只")
    print(f"{'='*60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C2 主版策略脚本（当日收盘买入版本）")
    parser.add_argument("--symbol", help="股票代码，如 600547（单股回测模式）")
    parser.add_argument("--bars", type=int, default=1200, help="获取日线条数")
    parser.add_argument("--out-trades", default="c2_trades.csv", help="交易明细输出 CSV")
    parser.add_argument("--out-summary", default="c2_summary.csv", help="汇总输出 CSV")
    parser.add_argument("--out-html", default="", help="可选 HTML 输出路径")
    parser.add_argument("--select-date", help="选股日期，格式 YYYY-MM-DD（选股模式）")
    parser.add_argument("--max-stocks", type=int, default=None, help="选股模式：最大扫描股票数量（默认全部）")
    parser.add_argument("--backfill", help="回补模式：开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", help="回补模式：结束日期，格式 YYYY-MM-DD（默认今天）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig()

    # 回补模式
    if args.backfill:
        backfill_selections(args.backfill, args.end_date, cfg, args.max_stocks)
        return

    # 选股模式
    if args.select_date:
        print(f"\n{'='*60}")
        print(f"C2 策略选股模式 - 日期: {args.select_date}")
        print(f"{'='*60}")
        print(f"选股条件:")
        print(f"  - 周线 DSA 位置 < {cfg.weekly_pos_thr}")
        print(f"  - 日线 DSA 位置 <= {cfg.dsa_thr}")
        print(f"  - VWAP 偏离度 <= {cfg.vwap_thr}%")
        print(f"  - Rope 方向 = 1（上升）")
        print(f"  - 趋势转变后 bars <= {cfg.bars_thr}")
        print(f"  - range_break_up = 1（向上突破）")
        print(f"{'='*60}\n")

        selection = scan_stocks_for_signals(args.select_date, cfg, args.max_stocks)

        if not selection.empty:
            saved_count = save_selection_to_db(selection, args.select_date, cfg)
            print(f"\n选股结果已保存到数据库，共 {saved_count} 只股票")
            print("\n选股结果预览:")
            print(selection.to_string(index=False))
        else:
            print(f"\n在 {args.select_date} 未找到满足条件的股票")
        return

    # 单股回测模式
    if not args.symbol:
        print("错误: 请指定 --symbol 进行单股回测，或指定 --select-date 进行选股，或指定 --backfill 进行回补")
        return

    engine = FactorEngine()

    # 从数据库加载数据（与选股模式一致）
    # 计算数据范围：从当前日期往前推 args.bars 个交易日
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=args.bars * 2)).strftime("%Y-%m-%d")

    # 将 symbol 转换为 ts_code 格式（如 600547 -> 600547.SH）
    ts_code = args.symbol if "." in args.symbol else (f"{args.symbol}.SH" if args.symbol.startswith("6") else f"{args.symbol}.SZ")

    raw = load_daily_from_db(ts_code, start_date, end_date)
    if raw.empty:
        print(f"错误: 无法从数据库获取 {ts_code} 的数据")
        return

    # 限制数据条数
    if len(raw) > args.bars:
        raw = raw.iloc[-args.bars:]

    raw["symbol"] = ts_code
    daily = engine.compute_daily_factors(raw)
    weekly = engine.compute_weekly_strict_dsa(raw, cfg.weekly_resample_rule)
    df = pd.concat([daily, weekly], axis=1)

    trades = generate_trades(df, cfg)
    summary = summarize_trades(trades, ts_code, cfg)

    trades.to_csv(args.out_trades, index=False)
    summary.to_csv(args.out_summary, index=False)

    if args.out_html:
        build_html(df, trades, args.out_html, ts_code)

    print(f"交易明细已保存: {args.out_trades}")
    print(f"策略汇总已保存: {args.out_summary}")
    if args.out_html:
        print(f"HTML 已保存: {args.out_html}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
