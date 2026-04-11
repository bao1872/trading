# -*- coding: utf-8 -*-
"""
日线底背离买点质量研究脚本（单 Excel 导出）

Purpose
- 仅研究日线周期。
- 复用 zigzag_macd_divergence_viewer 中的核心计算函数，直接以 zigzag + MACD 底背离作为买入触发事件。
- 围绕每个底背离事件，切出对应的上涨段 L1->H1 和回踩段 H1->L2。
- 提取上涨段质量、回踩段质量、trend 方向质量、事件当天质量，以及成交量 zscore（含 vol_z_5）。
- 输出单个 Excel，包含 README / dataset_summary / structure_table / event_table / label_table / event_quality_summary / bucket_profiles / feature_inventory。

研究约定
- 周期固定为日线（freq=d）。
- 每个确认的底背离都保留；同一结构内若连续出现多个底背离，不去重。
- 买入时点统一为：信号确认后的下一交易日开盘。
- 当前脚本只研究买点质量，不做卖点、不做完整交易回测净值。

How to Run
    python daily_bottom_div_quality_explorer.py --n-stocks 200 --bars 1000 --freq d \
        --excel-name daily_bottom_div_quality_report.xlsx

Examples
    python daily_bottom_div_quality_explorer.py --n-stocks 100 --bars 800 --freq d
    python daily_bottom_div_quality_explorer.py --n-stocks 300 --bars 1200 --freq d --refresh-cache

Dependencies
    pip install pandas numpy openpyxl sqlalchemy pytdx plotly

Outputs
- 一个 Excel 文件，默认输出到 daily_bottom_div_quality_output/ 目录下。

Notes
- 需要可用的数据库环境（复用现有 datasource.database / stock_k_data / stock_pools）。
- 需要本地可访问 zigzag_macd_divergence_viewer(1).py 或同名模块，以复用 compute_zigzag_macd。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score
except Exception:  # pragma: no cover
    GradientBoostingClassifier = None  # type: ignore[assignment]
    GradientBoostingRegressor = None  # type: ignore[assignment]
    SimpleImputer = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]

OUT_DIR = "daily_bottom_div_quality_output"
os.makedirs(OUT_DIR, exist_ok=True)

RET_WINDOWS = [3, 5, 10, 20]
ENTRY_LAG = 1
ENTRY_PRICE_MODE = "next_open"
DEFAULT_VOL_Z_WINDOWS = (5, 10, 20, 40)
RANDOM_STATE = 42


# =========================
# Feature governance
# =========================
DROP_EVENT_FEATURES = {
    "sentiment_code", "price_pivot_dir", "osc_pivot_dir", "pivot_price", "pivot_bar",
    "trigger_px", "trigger_macd", "trigger_signal", "trigger_di_plus", "trigger_di_minus", "trigger_dx",
    "bars_from_H1_to_trigger", "bars_from_L2_to_trigger", "trigger_position_in_pb",
    "trigger_di_plus_gt_di_minus_flag", "event_idx",
}

DROP_STRUCTURE_FEATURES = {
    "L1_px", "H1_px", "L2_px", "L1_idx", "H1_idx", "L2_idx",
    "up_return_per_bar", "up_max_drawdown_inside_leg", "up_macd_mean", "up_hist_pos_area", "up_hist_pos_ratio",
    "up_trend_pos_ratio", "up_vol_z_mean_10", "up_vol_z_mean_40", "up_vol_z_max_5", "up_vol_z_max_10",
    "up_vol_z_max_20", "up_vol_z_max_40", "up_adx_max", "up_dx_mean", "up_di_plus_mean", "up_di_minus_mean",
    "up_di_plus_dominance_ratio", "up_di_minus_dominance_ratio",
    "pb_efficiency", "pb_macd_mean", "pb_hist_neg_ratio", "pb_hist_min", "pb_trend_pos_ratio",
    "pb_trend_flip_count", "pb_trend_stable_flag", "pb_vol_z_mean_5", "pb_vol_z_mean_10", "pb_vol_z_mean_40",
    "pb_vol_z_max_5", "pb_vol_z_max_10", "pb_vol_z_max_40", "pb_vol_z_min_5", "pb_vol_z_min_10",
    "pb_vol_z_min_20", "pb_vol_z_min_40", "pb_low_bar_vol_z_5", "pb_low_bar_vol_z_10",
    "pb_low_bar_vol_z_20", "pb_low_bar_vol_z_40", "pb_adx_max", "pb_dx_mean", "pb_di_plus_mean",
    "pb_di_minus_mean", "pb_di_plus_dominance_ratio", "pb_di_minus_dominance_ratio",
}

KEEP_EVENT_FEATURES = [
    "symbol", "structure_id", "event_id", "trigger_dt",
    "divergence", "divergence_id", "bias", "supertrend_dir",
    "trigger_ret_1d", "trigger_hist",
    "trigger_vol_z_5", "trigger_vol_z_20",
    "trigger_adx", "trigger_di_spread", "trigger_adx_above_20_flag",
    "trend_preserved_during_pullback_flag", "pb_vol_z_mean_20_vs_up",
]

KEEP_STRUCTURE_FEATURES = [
    "symbol", "structure_id", "L1_dt", "H1_dt", "L2_dt",
    "up_return", "up_bars", "up_hist_mean",
    "up_adx_mean", "up_di_spread_mean", "up_trend_flip_count", "up_trend_stable_flag",
    "pb_return", "pb_bars", "pb_depth_vs_up", "pb_bars_vs_up",
    "pb_hist_mean", "pb_hist_neg_area",
    "pb_adx_mean", "pb_di_spread_mean",
    "pb_trend_neg_ratio", "pb_vol_z_mean_20", "pb_vol_z_max_20",
]

KEEP_LABEL_FEATURES = [
    "symbol", "event_id", "structure_id", "entry_dt",
    "ret_3", "ret_5", "ret_10", "ret_20",
    "mfe_10", "mae_10", "rr_10", "mfe_20", "mae_20", "rr_20",
]

# Core summary features used by bucket profiling and compact reporting.
# Keep this as a single source of truth so later refactors do not leave stale names behind.
CORE_FEATURES_FOR_SUMMARY = set(
    KEEP_EVENT_FEATURES + KEEP_STRUCTURE_FEATURES
) - {"symbol", "structure_id", "event_id", "trigger_dt", "L1_dt", "H1_dt", "L2_dt"}

DEFAULT_BUCKET_FEATURES = [
    "pb_depth_vs_up", "pb_bars_vs_up", "pb_vol_z_mean_20_vs_up",
    "trigger_vol_z_5", "trigger_vol_z_20",
    "trigger_adx", "trigger_di_spread", "trigger_adx_above_20_flag",
    "up_adx_mean", "pb_adx_mean", "up_di_spread_mean", "pb_di_spread_mean",
]


# =========================
# Dynamic imports / environment
# =========================
def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    # 先注册到sys.modules，避免dataclass等装饰器找不到模块
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# database helpers
try:
    from datasource.database import get_engine
except Exception:
    # 运行环境若无该模块，后续会在 load_kline 时抛错
    get_engine = None  # type: ignore[assignment]

# zigzag core
_zigzag_module = None
for candidate in [
    os.path.join(CURRENT_DIR, "zigzag_macd_divergence_viewer.py"),
    os.path.join(CURRENT_DIR, "zigzag_macd_divergence_viewer(1).py"),
    os.path.join(CURRENT_DIR, "zigzag_macd_divergence_viewer(2).py"),
    os.path.join(CURRENT_DIR, "zigzag_macd_divergence_viewer(3).py"),
    os.path.join(PARENT_DIR, "zigzag_macd_divergence_viewer.py"),
    os.path.join(PARENT_DIR, "zigzag_macd_divergence_viewer(1).py"),
    os.path.join(PARENT_DIR, "zigzag_macd_divergence_viewer(2).py"),
    os.path.join(PARENT_DIR, "zigzag_macd_divergence_viewer(3).py"),
    os.path.join(PARENT_DIR, "features", "zigzag_macd_divergence_viewer.py"),
    os.path.join(PARENT_DIR, "features", "zigzag_macd_divergence_viewer(1).py"),
    os.path.join(PARENT_DIR, "features", "zigzag_macd_divergence_viewer(2).py"),
    os.path.join(PARENT_DIR, "features", "zigzag_macd_divergence_viewer(3).py"),
]:
    if os.path.exists(candidate):
        _zigzag_module = _load_module_from_path("zigzag_macd_divergence_viewer_dyn", candidate)
        break
if _zigzag_module is None:
    try:
        import zigzag_macd_divergence_viewer as _zigzag_module  # type: ignore
    except Exception as exc:
        raise ImportError("无法加载 zigzag_macd_divergence_viewer 脚本，请将其放在当前目录或上级目录") from exc

ZigzagMacdConfig = _zigzag_module.ZigzagMacdConfig
compute_zigzag_macd = _zigzag_module.compute_zigzag_macd


# =========================
# Financial Score Integration
# =========================
FINANCIAL_SCORE_TABLE = "stock_financial_score_pool"


def get_financial_score_as_factor(
    ts_code: str,
    trade_date: pd.Timestamp,
    engine
) -> Dict[str, float]:
    """
    获取指定交易日有效的财务评分

    逻辑:
    1. 查询该股票所有已公告的季度评分
    2. 筛选 report_date <= trade_date 的最新记录
    3. 返回 total_score 和各维度评分

    注意: 由于表中缺少 ann_date，使用 report_date 作为替代，
          这可能导致一定的未来函数风险（实际公告通常晚于报告期）
    """
    from sqlalchemy import text

    trade_date_str = pd.to_datetime(trade_date).strftime("%Y%m%d")

    sql = """
    SELECT report_date, total_score,
           规模与增长_score, 盈利能力_score, 利润质量_score,
           现金创造能力_score, 资产效率与资金占用_score, 边际变化与持续性_score
    FROM stock_financial_score_pool
    WHERE ts_code = :ts_code
      AND report_date <= :trade_date
    ORDER BY report_date DESC
    LIMIT 1
    """
    try:
        result = pd.read_sql(
            text(sql),
            engine,
            params={"ts_code": ts_code, "trade_date": trade_date_str}
        )
    except Exception:
        # 如果查询失败（如表不存在），返回空字典
        return {}

    if result.empty:
        return {}

    row = result.iloc[0]
    return {
        "fin_total_score": float(row["total_score"]) if pd.notna(row["total_score"]) else np.nan,
        "fin_growth_score": float(row["规模与增长_score"]) if pd.notna(row["规模与增长_score"]) else np.nan,
        "fin_profit_score": float(row["盈利能力_score"]) if pd.notna(row["盈利能力_score"]) else np.nan,
        "fin_quality_score": float(row["利润质量_score"]) if pd.notna(row["利润质量_score"]) else np.nan,
        "fin_cash_score": float(row["现金创造能力_score"]) if pd.notna(row["现金创造能力_score"]) else np.nan,
        "fin_efficiency_score": float(row["资产效率与资金占用_score"]) if pd.notna(row["资产效率与资金占用_score"]) else np.nan,
        "fin_momentum_score": float(row["边际变化与持续性_score"]) if pd.notna(row["边际变化与持续性_score"]) else np.nan,
        "fin_report_date": str(row["report_date"]) if pd.notna(row["report_date"]) else None,
    }


def add_financial_score_factors(event_table: pd.DataFrame, engine=None) -> pd.DataFrame:
    """
    为底背离事件表添加财务评分因子

    新增字段:
    - fin_total_score: 综合财务评分
    - fin_growth_score: 规模与增长评分
    - fin_profit_score: 盈利能力评分
    - fin_quality_score: 利润质量评分
    - fin_cash_score: 现金创造能力评分
    - fin_efficiency_score: 资产效率评分
    - fin_momentum_score: 边际变化与持续性评分
    - fin_report_date: 使用的报告期
    """
    if event_table.empty:
        return event_table

    if engine is None:
        engine = get_engine()

    if engine is None:
        # 无法连接数据库，返回原表（添加空列）
        for col in ["fin_total_score", "fin_growth_score", "fin_profit_score",
                    "fin_quality_score", "fin_cash_score", "fin_efficiency_score",
                    "fin_momentum_score", "fin_report_date"]:
            event_table[col] = np.nan
        return event_table

    scores = []
    for _, row in tqdm(event_table.iterrows(), total=len(event_table), desc="加载财务评分"):
        score = get_financial_score_as_factor(
            row['symbol'],
            pd.to_datetime(row['trigger_dt']),
            engine
        )
        scores.append(score)

    # 合并到事件表
    score_df = pd.DataFrame(scores)
    result = pd.concat([event_table.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
    return result


# =========================
# Utilities
# =========================
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def rr_from_ret_dd(ret: float, dd: float) -> float:
    if pd.isna(ret) or pd.isna(dd) or dd == 0:
        return np.nan
    return float(ret / abs(dd))


def _sanitize_sheet_name(name: str, used: set[str]) -> str:
    bad = [":", "\\", "/", "?", "*", "[", "]"]
    for ch in bad:
        name = name.replace(ch, "_")
    name = name[:31]
    base = name
    idx = 1
    while name in used:
        suffix = f"_{idx}"
        name = (base[: 31 - len(suffix)] + suffix)[:31]
        idx += 1
    used.add(name)
    return name


def build_readme_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "项目": [
                "研究目标",
                "周期",
                "买入触发",
                "买入价定义",
                "样本保留规则",
                "结构定义",
                "输出表",
            ],
            "说明": [
                "评估日线底背离在当时作为买点的质量，而不是做去重后的策略收益回测。",
                "固定日线 freq=d。",
                "zigzag + MACD 底背离事件。",
                "信号确认后下一交易日开盘价。",
                "同一结构内连续多个底背离全部保留，每个都作为独立买点样本。",
                "围绕事件向前切出 L1->H1 上涨段与 H1->L2 回踩段。",
                "README / dataset_summary / event_quality_summary / yearly_summary / type_year_summary / bucket_profiles / feature_keep_drop_report / feature_inventory",
            ],
        }
    )


def export_results_workbook(result_tables: Dict[str, pd.DataFrame], out_dir: str, workbook_name: str) -> str:
    workbook_path = os.path.join(out_dir, workbook_name)
    used: set[str] = set()
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        build_readme_sheet().to_excel(writer, sheet_name=_sanitize_sheet_name("README", used), index=False)
        for sheet, df in result_tables.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=_sanitize_sheet_name(sheet, used), index=False)
    return workbook_path


# =========================
# Data loaders (aligned to existing explorer style)
# =========================
def load_kline(ts_code: str, freq: str = "d", bars: int = 1000) -> Optional[pd.DataFrame]:
    if get_engine is None:
        raise RuntimeError("缺少 datasource.database.get_engine，无法读取数据库")
    from sqlalchemy import text

    engine = get_engine()
    sql = """
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
        ORDER BY bar_time DESC LIMIT :limit
    """
    try:
        df = pd.read_sql_query(text(sql), engine, params={"ts_code": ts_code, "freq": freq, "limit": bars})
    except Exception as exc:
        print(f"  [WARN] {ts_code} 查询失败: {exc}")
        return None
    finally:
        engine.dispose()
    if df.empty:
        return None
    df = df.sort_values("bar_time").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["bar_time"]).dt.tz_localize(None)
    df = df.set_index("datetime")
    df["vol"] = df["volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)
    return df[["open", "high", "low", "close", "vol"]].astype(float)


def get_stock_pool(n: int = 100, seed: int = 42) -> List[str]:
    if get_engine is None:
        raise RuntimeError("缺少 datasource.database.get_engine，无法读取数据库")
    from sqlalchemy import text

    engine = get_engine()
    sql = "SELECT ts_code FROM stock_pools ORDER BY ts_code"
    try:
        df = pd.read_sql_query(text(sql), engine)
    finally:
        engine.dispose()
    if df.empty:
        raise ValueError("stock_pools表为空")
    codes = df["ts_code"].tolist()
    rng = np.random.default_rng(seed)
    return rng.choice(codes, size=min(n, len(codes)), replace=False).tolist()


# =========================
# Cache helpers
# =========================
def get_cache_path(cache_dir: str, n_stocks: int, bars: int, freq: str, seed: int) -> str:
    param_str = f"bottom_div_{n_stocks}_{bars}_{freq}_{seed}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    cache_file = f"bottom_div_{freq}_{bars}_{n_stocks}stocks_{param_hash}.pkl"
    return os.path.join(cache_dir, cache_file)


def cache_exists(cache_path: str) -> bool:
    parquet_path = cache_path.replace(".pkl", ".parquet")
    return os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0


def save_cache(data: Dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    all_dfs = data.get("all_dfs", [])
    metadata = data.get("metadata", {})
    parquet_path = cache_path.replace(".pkl", ".parquet")
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_parquet(parquet_path, index=False, compression="zstd")
    meta_path = cache_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_cache(cache_path: str) -> Optional[Dict]:
    try:
        parquet_path = cache_path.replace(".pkl", ".parquet")
        meta_path = cache_path.replace(".pkl", "_meta.json")
        if not os.path.exists(parquet_path):
            return None
        combined_df = pd.read_parquet(parquet_path)
        all_dfs = []
        for symbol in combined_df["symbol"].unique():
            df = combined_df[combined_df["symbol"] == symbol].copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            all_dfs.append(df)
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        return {"all_dfs": all_dfs, "metadata": metadata}
    except Exception as exc:
        print(f"  [WARN] 缓存加载失败: {exc}")
        return None


# =========================
# Feature engineering
# =========================
def add_volume_zscores(df: pd.DataFrame, windows: Sequence[int] = DEFAULT_VOL_Z_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    vol = safe_numeric(out["vol"])
    for w in windows:
        ma = vol.rolling(w, min_periods=max(3, w // 2)).mean()
        std = vol.rolling(w, min_periods=max(3, w // 2)).std(ddof=0).replace(0, np.nan)
        out[f"vol_ma_{w}"] = ma
        out[f"vol_std_{w}"] = std
        out[f"vol_z_{w}"] = (vol - ma) / std
    return out


def add_event_day_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    open_ = safe_numeric(out["open"])
    high = safe_numeric(out["high"])
    low = safe_numeric(out["low"])
    close = safe_numeric(out["close"])
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    out["trigger_ret_1d_raw"] = close.pct_change(1)
    out["body_pct_raw"] = body / rng
    out["upper_shadow_pct_raw"] = (high - np.maximum(open_, close)) / rng
    out["lower_shadow_pct_raw"] = (np.minimum(open_, close) - low) / rng
    out["close_pos_raw"] = (close - low) / rng
    return out


def compute_base_features(df: pd.DataFrame, zigzag_cfg: ZigzagMacdConfig, vol_z_windows: Sequence[int]) -> pd.DataFrame:
    base = df.copy()
    zz_df, _, _, _ = compute_zigzag_macd(base, zigzag_cfg)
    merged = pd.concat([base, zz_df], axis=1)
    merged = add_volume_zscores(merged, windows=vol_z_windows)
    merged = add_event_day_price_features(merged)
    merged["bar_index"] = np.arange(len(merged), dtype=int)
    return merged


# =========================
# Structure / event extraction
# =========================
def extract_pivot_sequence(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pivot_price",
        "pivot_bar",
        "price_pivot_dir",
        "osc_pivot_dir",
        "divergence",
        "divergence_id",
        "bias",
        "sentiment_code",
        "supertrend_dir",
        "macd_line",
        "macd_signal",
        "macd_hist",
    ]
    piv = df.loc[df["pivot_price"].notna(), cols].copy()
    piv = piv.reset_index()
    # 兼容索引有名或无名的情况
    index_col = "index" if "index" in piv.columns else piv.columns[0]
    piv = piv.rename(columns={index_col: "pivot_dt"})
    piv["pivot_idx"] = piv["pivot_bar"].astype("Int64")
    piv = piv.dropna(subset=["pivot_idx"]).copy()
    piv["pivot_idx"] = piv["pivot_idx"].astype(int)
    piv = piv.drop_duplicates(subset=["pivot_idx", "divergence_id"], keep="last")
    piv = piv.sort_values("pivot_idx").reset_index(drop=True)
    return piv


def is_bottom_divergence(row: pd.Series) -> bool:
    # 以低点 pivot 为前提，且存在非零 divergence。这样把所有底背离样本都保留下来供质量研究。
    pdir = row.get("price_pivot_dir")
    div = row.get("divergence")
    return pd.notna(pdir) and pd.notna(div) and float(pdir) < 0 and float(div) != 0.0


def find_structure_for_event(pivots: pd.DataFrame, event_pivot_idx: int) -> Optional[Dict]:
    piv = pivots[pivots["pivot_idx"] <= event_pivot_idx].sort_values("pivot_idx")
    if piv.empty:
        return None
    lows = piv[piv["price_pivot_dir"] < 0]
    highs = piv[piv["price_pivot_dir"] > 0]
    if lows.empty:
        return None

    l2_row = lows.iloc[-1]
    highs_before_l2 = highs[highs["pivot_idx"] < int(l2_row["pivot_idx"])]
    if highs_before_l2.empty:
        return None
    h1_row = highs_before_l2.iloc[-1]
    lows_before_h1 = lows[lows["pivot_idx"] < int(h1_row["pivot_idx"])]
    if lows_before_h1.empty:
        return None
    l1_row = lows_before_h1.iloc[-1]

    return {
        "L1_idx": int(l1_row["pivot_idx"]),
        "L1_dt": l1_row["pivot_dt"],
        "L1_px": float(l1_row["pivot_price"]),
        "H1_idx": int(h1_row["pivot_idx"]),
        "H1_dt": h1_row["pivot_dt"],
        "H1_px": float(h1_row["pivot_price"]),
        "L2_idx": int(l2_row["pivot_idx"]),
        "L2_dt": l2_row["pivot_dt"],
        "L2_px": float(l2_row["pivot_price"]),
    }


def _net_to_total_move(close: np.ndarray) -> float:
    if len(close) < 2:
        return np.nan
    net = close[-1] - close[0]
    total = np.sum(np.abs(np.diff(close)))
    if total == 0:
        return np.nan
    return float(net / total)


def _trend_flip_count(series: pd.Series) -> float:
    s = safe_numeric(series).dropna()
    if len(s) <= 1:
        return 0.0
    return float((s != s.shift(1)).sum() - 1)


def _seg_series(seg: pd.DataFrame, col: str) -> pd.Series:
    return safe_numeric(seg[col]) if col in seg.columns else pd.Series(dtype=float)


def add_adx_di_features(seg: pd.DataFrame, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    adx = _seg_series(seg, "adx")
    di_plus = _seg_series(seg, "di_plus")
    di_minus = _seg_series(seg, "di_minus")
    dx = _seg_series(seg, "dx")
    spread = di_plus - di_minus if len(di_plus) and len(di_minus) else pd.Series(dtype=float)

    out[f"{prefix}_adx_mean"] = float(adx.mean()) if adx.notna().any() else np.nan
    out[f"{prefix}_adx_max"] = float(adx.max()) if adx.notna().any() else np.nan
    out[f"{prefix}_adx_above_20_ratio"] = float((adx > 20).mean()) if adx.notna().any() else np.nan
    out[f"{prefix}_dx_mean"] = float(dx.mean()) if dx.notna().any() else np.nan
    out[f"{prefix}_di_plus_mean"] = float(di_plus.mean()) if di_plus.notna().any() else np.nan
    out[f"{prefix}_di_minus_mean"] = float(di_minus.mean()) if di_minus.notna().any() else np.nan
    out[f"{prefix}_di_spread_mean"] = float(spread.mean()) if len(spread) and spread.notna().any() else np.nan
    out[f"{prefix}_di_plus_dominance_ratio"] = float((di_plus > di_minus).mean()) if di_plus.notna().any() and di_minus.notna().any() else np.nan
    out[f"{prefix}_di_minus_dominance_ratio"] = float((di_minus > di_plus).mean()) if di_plus.notna().any() and di_minus.notna().any() else np.nan
    return out


def calc_up_leg_features(df: pd.DataFrame, l1_idx: int, h1_idx: int) -> Dict[str, float]:
    seg = df.iloc[l1_idx:h1_idx + 1].copy()
    close = safe_numeric(seg["close"]).to_numpy(float)
    out: Dict[str, float] = {}
    if len(seg) < 2:
        return out
    out["up_return"] = float(close[-1] / close[0] - 1.0) if close[0] > 0 else np.nan
    out["up_bars"] = float(len(seg) - 1)
    out["up_return_per_bar"] = out["up_return"] / max(out["up_bars"], 1.0) if pd.notna(out["up_return"]) else np.nan
    out["up_efficiency"] = _net_to_total_move(close)
    running_max = np.maximum.accumulate(close)
    drawdowns = close / np.where(running_max == 0, np.nan, running_max) - 1.0
    out["up_max_drawdown_inside_leg"] = float(np.nanmin(drawdowns)) if len(drawdowns) else np.nan

    hist = safe_numeric(seg["macd_hist"])
    macd = safe_numeric(seg["macd_line"])
    out["up_macd_mean"] = float(macd.mean()) if macd.notna().any() else np.nan
    out["up_hist_mean"] = float(hist.mean()) if hist.notna().any() else np.nan
    out["up_hist_pos_area"] = float(hist.clip(lower=0).sum()) if hist.notna().any() else np.nan
    out["up_hist_pos_ratio"] = float((hist > 0).mean()) if hist.notna().any() else np.nan

    trend = safe_numeric(seg["supertrend_dir"])
    out["up_trend_pos_ratio"] = float((trend > 0).mean()) if trend.notna().any() else np.nan
    out["up_trend_flip_count"] = _trend_flip_count(trend)
    out["up_trend_stable_flag"] = float((trend > 0).all()) if trend.notna().any() else np.nan

    for w in DEFAULT_VOL_Z_WINDOWS:
        col = f"vol_z_{w}"
        z = safe_numeric(seg[col]) if col in seg.columns else pd.Series(dtype=float)
        out[f"up_vol_z_mean_{w}"] = float(z.mean()) if len(z) and z.notna().any() else np.nan
        out[f"up_vol_z_max_{w}"] = float(z.max()) if len(z) and z.notna().any() else np.nan
    out.update(add_adx_di_features(seg, "up"))
    return out


def calc_pullback_features(df: pd.DataFrame, h1_idx: int, l2_idx: int, up_features: Dict[str, float]) -> Dict[str, float]:
    seg = df.iloc[h1_idx:l2_idx + 1].copy()
    close = safe_numeric(seg["close"]).to_numpy(float)
    out: Dict[str, float] = {}
    if len(seg) < 2:
        return out
    out["pb_return"] = float(close[-1] / close[0] - 1.0) if close[0] > 0 else np.nan
    out["pb_bars"] = float(len(seg) - 1)
    up_ret = up_features.get("up_return")
    up_bars = up_features.get("up_bars")
    out["pb_depth_vs_up"] = abs(out["pb_return"]) / abs(up_ret) if pd.notna(out["pb_return"]) and pd.notna(up_ret) and up_ret != 0 else np.nan
    out["pb_bars_vs_up"] = out["pb_bars"] / up_bars if pd.notna(out["pb_bars"]) and pd.notna(up_bars) and up_bars != 0 else np.nan
    out["pb_efficiency"] = _net_to_total_move(close)

    hist = safe_numeric(seg["macd_hist"])
    macd = safe_numeric(seg["macd_line"])
    out["pb_macd_mean"] = float(macd.mean()) if macd.notna().any() else np.nan
    out["pb_hist_mean"] = float(hist.mean()) if hist.notna().any() else np.nan
    out["pb_hist_neg_area"] = float((-hist.clip(upper=0)).sum()) if hist.notna().any() else np.nan
    out["pb_hist_neg_ratio"] = float((hist < 0).mean()) if hist.notna().any() else np.nan
    out["pb_hist_min"] = float(hist.min()) if hist.notna().any() else np.nan

    trend = safe_numeric(seg["supertrend_dir"])
    out["pb_trend_pos_ratio"] = float((trend > 0).mean()) if trend.notna().any() else np.nan
    out["pb_trend_neg_ratio"] = float((trend < 0).mean()) if trend.notna().any() else np.nan
    out["pb_trend_flip_count"] = _trend_flip_count(trend)
    out["pb_trend_stable_flag"] = float((trend > 0).all()) if trend.notna().any() else np.nan

    for w in DEFAULT_VOL_Z_WINDOWS:
        col = f"vol_z_{w}"
        z = safe_numeric(seg[col]) if col in seg.columns else pd.Series(dtype=float)
        out[f"pb_vol_z_mean_{w}"] = float(z.mean()) if len(z) and z.notna().any() else np.nan
        out[f"pb_vol_z_max_{w}"] = float(z.max()) if len(z) and z.notna().any() else np.nan
        out[f"pb_vol_z_min_{w}"] = float(z.min()) if len(z) and z.notna().any() else np.nan
    out.update(add_adx_di_features(seg, "pb"))
    # L2 当天量能
    for w in DEFAULT_VOL_Z_WINDOWS:
        col = f"vol_z_{w}"
        out[f"pb_low_bar_vol_z_{w}"] = float(df.iloc[l2_idx][col]) if col in df.columns and pd.notna(df.iloc[l2_idx][col]) else np.nan
    return out


def calc_event_features(df: pd.DataFrame, event_idx: int, h1_idx: int, l2_idx: int) -> Dict[str, float]:
    row = df.iloc[event_idx]
    out: Dict[str, float] = {
        "event_idx": int(event_idx),
        "trigger_dt": df.index[event_idx],
        "trigger_px": float(df.iloc[event_idx + ENTRY_LAG]["open"]) if event_idx + ENTRY_LAG < len(df) else np.nan,
        "divergence": float(row["divergence"]) if pd.notna(row["divergence"]) else np.nan,
        "divergence_id": row["divergence_id"] if pd.notna(row["divergence_id"]) else "",
        "bias": float(row["bias"]) if pd.notna(row["bias"]) else np.nan,
        "sentiment_code": float(row["sentiment_code"]) if pd.notna(row["sentiment_code"]) else np.nan,
        "supertrend_dir": float(row["supertrend_dir"]) if pd.notna(row["supertrend_dir"]) else np.nan,
        "price_pivot_dir": float(row["price_pivot_dir"]) if pd.notna(row["price_pivot_dir"]) else np.nan,
        "osc_pivot_dir": float(row["osc_pivot_dir"]) if pd.notna(row["osc_pivot_dir"]) else np.nan,
        "pivot_price": float(row["pivot_price"]) if pd.notna(row["pivot_price"]) else np.nan,
        "pivot_bar": float(row["pivot_bar"]) if pd.notna(row["pivot_bar"]) else np.nan,
        "trigger_ret_1d": float(row.get("trigger_ret_1d_raw", np.nan)),
        "trigger_body_pct": float(row.get("body_pct_raw", np.nan)),
        "trigger_upper_shadow_pct": float(row.get("upper_shadow_pct_raw", np.nan)),
        "trigger_lower_shadow_pct": float(row.get("lower_shadow_pct_raw", np.nan)),
        "trigger_close_pos": float(row.get("close_pos_raw", np.nan)),
        "trigger_macd": float(row["macd_line"]) if pd.notna(row["macd_line"]) else np.nan,
        "trigger_signal": float(row["macd_signal"]) if pd.notna(row["macd_signal"]) else np.nan,
        "trigger_hist": float(row["macd_hist"]) if pd.notna(row["macd_hist"]) else np.nan,
        "trigger_adx": float(row["adx"]) if "adx" in row.index and pd.notna(row["adx"]) else np.nan,
        "trigger_di_plus": float(row["di_plus"]) if "di_plus" in row.index and pd.notna(row["di_plus"]) else np.nan,
        "trigger_di_minus": float(row["di_minus"]) if "di_minus" in row.index and pd.notna(row["di_minus"]) else np.nan,
        "trigger_dx": float(row["dx"]) if "dx" in row.index and pd.notna(row["dx"]) else np.nan,
        "bars_from_H1_to_trigger": float(event_idx - h1_idx),
        "bars_from_L2_to_trigger": float(event_idx - l2_idx),
        "trigger_position_in_pb": float((event_idx - h1_idx) / max(l2_idx - h1_idx, 1)),
    }
    out["trigger_di_spread"] = out["trigger_di_plus"] - out["trigger_di_minus"] if pd.notna(out["trigger_di_plus"]) and pd.notna(out["trigger_di_minus"]) else np.nan
    out["trigger_adx_above_20_flag"] = float(out["trigger_adx"] > 20) if pd.notna(out["trigger_adx"]) else np.nan
    out["trigger_di_plus_gt_di_minus_flag"] = float(out["trigger_di_plus"] > out["trigger_di_minus"]) if pd.notna(out["trigger_di_plus"]) and pd.notna(out["trigger_di_minus"]) else np.nan
    for w in DEFAULT_VOL_Z_WINDOWS:
        col = f"vol_z_{w}"
        out[f"trigger_vol_z_{w}"] = float(row[col]) if col in df.columns and pd.notna(row[col]) else np.nan
    return out


def calc_relation_features(up_features: Dict[str, float], pb_features: Dict[str, float], event_features: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    up_pos_area = up_features.get("up_hist_pos_area")
    pb_neg_area = pb_features.get("pb_hist_neg_area")
    out["pb_hist_neg_area_vs_up_hist_pos_area"] = (
        pb_neg_area / up_pos_area if pd.notna(pb_neg_area) and pd.notna(up_pos_area) and up_pos_area != 0 else np.nan
    )
    out["pb_hist_mean_vs_up_hist_mean"] = (
        pb_features.get("pb_hist_mean") / up_features.get("up_hist_mean")
        if pd.notna(pb_features.get("pb_hist_mean")) and pd.notna(up_features.get("up_hist_mean")) and up_features.get("up_hist_mean") not in [0, None]
        else np.nan
    )
    out["pb_vol_z_mean_20_vs_up"] = (
        pb_features.get("pb_vol_z_mean_20") / up_features.get("up_vol_z_mean_20")
        if pd.notna(pb_features.get("pb_vol_z_mean_20")) and pd.notna(up_features.get("up_vol_z_mean_20")) and up_features.get("up_vol_z_mean_20") not in [0, None]
        else np.nan
    )
    out["trend_preserved_during_pullback_flag"] = float((pb_features.get("pb_trend_pos_ratio", np.nan) == 1.0)) if pd.notna(pb_features.get("pb_trend_pos_ratio")) else np.nan
    out["trigger_trend_same_as_up_flag"] = float(
        pd.notna(event_features.get("supertrend_dir"))
        and pd.notna(up_features.get("up_trend_pos_ratio"))
        and event_features.get("supertrend_dir") > 0
        and up_features.get("up_trend_pos_ratio") > 0.5
    )
    return out


def build_structures_and_events(symbol_df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivots = extract_pivot_sequence(symbol_df)
    events_rows: List[Dict] = []
    structure_map: Dict[Tuple[int, int, int], Dict] = {}

    for _, p in pivots.iterrows():
        if not is_bottom_divergence(p):
            continue
        event_idx = int(p["pivot_idx"])
        struct = find_structure_for_event(pivots, event_idx)
        if struct is None:
            continue
        key = (struct["L1_idx"], struct["H1_idx"], struct["L2_idx"])
        if key not in structure_map:
            up_features = calc_up_leg_features(symbol_df, struct["L1_idx"], struct["H1_idx"])
            pb_features = calc_pullback_features(symbol_df, struct["H1_idx"], struct["L2_idx"], up_features)
            structure_row = {
                "symbol": symbol,
                "structure_id": f"{symbol}_{struct['L1_idx']}_{struct['H1_idx']}_{struct['L2_idx']}",
                **struct,
                **up_features,
                **pb_features,
            }
            structure_map[key] = structure_row

        structure_row = structure_map[key]
        event_features = calc_event_features(symbol_df, event_idx, struct["H1_idx"], struct["L2_idx"])
        up_feature_view = {k: v for k, v in structure_row.items() if str(k).startswith("up_")}
        pb_feature_view = {k: v for k, v in structure_row.items() if str(k).startswith("pb_")}
        relation_features = calc_relation_features(up_feature_view, pb_feature_view, event_features)
        event_row = {
            "symbol": symbol,
            "structure_id": structure_row["structure_id"],
            "event_id": f"{symbol}_{event_idx}_{str(event_features.get('divergence_id',''))}",
            **event_features,
            **relation_features,
        }
        events_rows.append(event_row)

    structure_table = pd.DataFrame(list(structure_map.values()))
    event_table = pd.DataFrame(events_rows)
    return structure_table, event_table


# =========================
# Labels
# =========================
def add_event_labels(events_df: pd.DataFrame, full_df: pd.DataFrame, windows: Sequence[int] = RET_WINDOWS) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()
    out = events_df.copy()
    open_ = full_df["open"].to_numpy(float)
    close = full_df["close"].to_numpy(float)
    high = full_df["high"].to_numpy(float)
    low = full_df["low"].to_numpy(float)
    n = len(full_df)

    for w in windows:
        out[f"ret_{w}"] = np.nan
        out[f"mfe_{w}"] = np.nan
        out[f"mae_{w}"] = np.nan
        out[f"rr_{w}"] = np.nan

    for ridx, row in out.iterrows():
        signal_idx = int(row["event_idx"])
        entry_idx = signal_idx + ENTRY_LAG
        if entry_idx >= n:
            continue
        entry = open_[entry_idx] if ENTRY_PRICE_MODE == "next_open" else close[entry_idx]
        if not np.isfinite(entry) or entry <= 0:
            continue
        out.at[ridx, "entry_idx"] = entry_idx
        out.at[ridx, "entry_dt"] = full_df.index[entry_idx]
        out.at[ridx, "entry_price"] = entry

        for w in windows:
            end = entry_idx + w
            if end >= n:
                continue
            out.at[ridx, f"ret_{w}"] = (close[end] - entry) / entry
            future_high = np.nanmax(high[entry_idx + 1: end + 1]) if end > entry_idx else np.nan
            future_low = np.nanmin(low[entry_idx + 1: end + 1]) if end > entry_idx else np.nan
            mfe = (future_high - entry) / entry if np.isfinite(future_high) else np.nan
            mae = min(0.0, (future_low - entry) / entry) if np.isfinite(future_low) else np.nan
            out.at[ridx, f"mfe_{w}"] = mfe
            out.at[ridx, f"mae_{w}"] = mae
            out.at[ridx, f"rr_{w}"] = rr_from_ret_dd(mfe, mae)

    return out


# =========================
# Summary / buckets
# =========================
def _round_numeric_frame(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def prune_feature_tables(structure_df: pd.DataFrame, event_df: pd.DataFrame, label_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _apply(df: pd.DataFrame, keep: Sequence[str], drop: set[str]) -> pd.DataFrame:
        if df.empty:
            return df
        cols = [c for c in keep if c in df.columns and c not in drop]
        return _round_numeric_frame(df[cols].copy())

    return (
        _apply(structure_df, KEEP_STRUCTURE_FEATURES, DROP_STRUCTURE_FEATURES),
        _apply(event_df, KEEP_EVENT_FEATURES, DROP_EVENT_FEATURES),
        _apply(label_df, KEEP_LABEL_FEATURES, set()),
    )


def build_feature_keep_drop_report(structure_df: pd.DataFrame, event_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for col in sorted(set(structure_df.columns) | set(event_df.columns) | set(label_df.columns)):
        location = []
        if col in structure_df.columns:
            location.append('structure')
        if col in event_df.columns:
            location.append('event')
        if col in label_df.columns:
            location.append('label')
        status = 'keep'
        reason = 'core_feature'
        if col in DROP_STRUCTURE_FEATURES or col in DROP_EVENT_FEATURES:
            status = 'drop'
            reason = 'redundant_or_low_value'
        elif col in KEEP_STRUCTURE_FEATURES or col in KEEP_EVENT_FEATURES or col in KEEP_LABEL_FEATURES:
            status = 'keep'
            reason = 'selected_core_feature'
        elif col.startswith('ret_') or col.startswith('mfe_') or col.startswith('mae_') or col.startswith('rr_') or col == 'entry_dt':
            status = 'keep'
            reason = 'label_or_analysis_key'
        else:
            status = 'drop'
            reason = 'not_exported_to_keep_outputs_small'
        rows.append({
            'feature': col,
            'location': ','.join(location),
            'status': status,
            'reason': reason,
        })
    return pd.DataFrame(rows)


def build_yearly_summary(label_df: pd.DataFrame) -> pd.DataFrame:
    if label_df.empty or 'entry_dt' not in label_df.columns:
        return pd.DataFrame()
    tmp = label_df.copy()
    tmp['entry_dt'] = pd.to_datetime(tmp['entry_dt'], errors='coerce')
    tmp['entry_year'] = tmp['entry_dt'].dt.year
    tmp = tmp[tmp['entry_year'].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()
    out = tmp.groupby('entry_year', dropna=False).agg(
        n=('event_id', 'count'),
        ret_3_mean=('ret_3', 'mean'),
        ret_5_mean=('ret_5', 'mean'),
        ret_10_mean=('ret_10', 'mean'),
        ret_20_mean=('ret_20', 'mean'),
        mfe_10_mean=('mfe_10', 'mean'),
        mae_10_mean=('mae_10', 'mean'),
        rr_10_mean=('rr_10', 'mean'),
    ).reset_index().rename(columns={'entry_year': 'year'})
    return _round_numeric_frame(out)


def build_type_year_summary(label_df: pd.DataFrame) -> pd.DataFrame:
    need = {'entry_dt'}
    if label_df.empty or not need.issubset(label_df.columns):
        return pd.DataFrame()
    tmp = label_df.copy()
    tmp['entry_dt'] = pd.to_datetime(tmp['entry_dt'], errors='coerce')
    tmp['entry_year'] = tmp['entry_dt'].dt.year
    tmp = tmp[tmp['entry_year'].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()
    group_cols = ['entry_year'] + (["divergence_id"] if 'divergence_id' in tmp.columns else ["divergence"])
    extra = [c for c in ['supertrend_dir', 'bias'] if c in tmp.columns]
    group_cols.extend(extra)
    out = tmp.groupby(group_cols, dropna=False).agg(
        n=('event_id', 'count'),
        ret_5_mean=('ret_5', 'mean'),
        ret_10_mean=('ret_10', 'mean'),
        ret_20_mean=('ret_20', 'mean'),
        rr_10_mean=('rr_10', 'mean'),
    ).reset_index().rename(columns={'entry_year': 'year'})
    return _round_numeric_frame(out)


def build_analysis_table(structure_df: pd.DataFrame, event_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    if label_df.empty:
        return label_df.copy()
    out = label_df.copy()
    event_extra = [c for c in event_df.columns if c not in out.columns and c not in {"symbol"}]
    if event_extra:
        out = out.merge(event_df[["event_id", *event_extra]], on="event_id", how="left")
    structure_extra = [c for c in structure_df.columns if c not in out.columns and c not in {"symbol"}]
    if structure_extra:
        merge_cols = ["structure_id", *structure_extra]
        if "symbol" in structure_df.columns and "symbol" in out.columns and "symbol" not in merge_cols:
            merge_cols = ["symbol", *merge_cols]
            on_cols = ["symbol", "structure_id"]
        else:
            on_cols = ["structure_id"]
        out = out.merge(structure_df[merge_cols], on=on_cols, how="left")
    return _round_numeric_frame(out)


def _safe_q(series: pd.Series, q: float, fallback: float = np.nan) -> float:
    s = safe_numeric(series).dropna()
    if s.empty:
        return fallback
    return float(s.quantile(q))


def _resolve_divergence_label(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    if "divergence_id" in df.columns:
        s = df["divergence_id"].astype(str).replace({"nan": np.nan, "None": np.nan})
        if s.notna().any():
            return s.fillna("UNK")
    div = safe_numeric(df.get("divergence", pd.Series(np.nan, index=df.index if len(df) else None)))
    mapping = {1.0: "C", -1.0: "D", -2.0: "H", 0.0: "I"}
    return div.map(mapping).fillna("UNK").astype(str)


def build_rule_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "trigger_di_spread_lo": _safe_q(df.get("trigger_di_spread", pd.Series(dtype=float)), 0.45, 0.0),
        "trigger_di_spread_md": _safe_q(df.get("trigger_di_spread", pd.Series(dtype=float)), 0.55, 0.0),
        "trigger_di_spread_hi": _safe_q(df.get("trigger_di_spread", pd.Series(dtype=float)), 0.70, 0.0),
        "trigger_adx_md": _safe_q(df.get("trigger_adx", pd.Series(dtype=float)), 0.50, 20.0),
        "trigger_adx_hi": _safe_q(df.get("trigger_adx", pd.Series(dtype=float)), 0.65, 20.0),
        "trigger_vol_z_20_md": _safe_q(df.get("trigger_vol_z_20", pd.Series(dtype=float)), 0.50, 0.0),
        "trigger_vol_z_5_md": _safe_q(df.get("trigger_vol_z_5", pd.Series(dtype=float)), 0.50, 0.0),
        "pb_depth_loose_max": _safe_q(df.get("pb_depth_vs_up", pd.Series(dtype=float)), 0.80, np.inf),
        "pb_depth_mid_max": _safe_q(df.get("pb_depth_vs_up", pd.Series(dtype=float)), 0.68, np.inf),
        "pb_depth_strict_max": _safe_q(df.get("pb_depth_vs_up", pd.Series(dtype=float)), 0.50, np.inf),
        "pb_bars_mid_max": _safe_q(df.get("pb_bars_vs_up", pd.Series(dtype=float)), 0.68, np.inf),
        "pb_bars_strict_max": _safe_q(df.get("pb_bars_vs_up", pd.Series(dtype=float)), 0.50, np.inf),
        "pb_hist_neg_mid_max": _safe_q(df.get("pb_hist_neg_area", pd.Series(dtype=float)), 0.68, np.inf),
        "pb_hist_neg_strict_max": _safe_q(df.get("pb_hist_neg_area", pd.Series(dtype=float)), 0.50, np.inf),
        "pb_vol_ratio_mid_max": _safe_q(df.get("pb_vol_z_mean_20_vs_up", pd.Series(dtype=float)), 0.72, np.inf),
        "pb_vol_ratio_strict_max": _safe_q(df.get("pb_vol_z_mean_20_vs_up", pd.Series(dtype=float)), 0.55, np.inf),
        "pb_trend_neg_ratio_soft_max": _safe_q(df.get("pb_trend_neg_ratio", pd.Series(dtype=float)), 0.75, 1.0),
        "pb_di_spread_mean_soft_min": _safe_q(df.get("pb_di_spread_mean", pd.Series(dtype=float)), 0.35, -np.inf),
    }


def build_rule_components(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, float]]:
    t = build_rule_thresholds(df)
    n = len(df)
    idx = df.index
    div_label = _resolve_divergence_label(df)
    bias = safe_numeric(df.get("bias", pd.Series(np.nan, index=idx)))
    st_dir = safe_numeric(df.get("supertrend_dir", pd.Series(np.nan, index=idx)))
    strict_trend_ok = safe_numeric(df.get("trend_preserved_during_pullback_flag", pd.Series(np.nan, index=idx))) == 1
    pb_trend_neg_ratio = safe_numeric(df.get("pb_trend_neg_ratio", pd.Series(np.nan, index=idx)))
    pb_di_spread_mean = safe_numeric(df.get("pb_di_spread_mean", pd.Series(np.nan, index=idx)))
    soft_trend_ok = strict_trend_ok | (pb_trend_neg_ratio <= t["pb_trend_neg_ratio_soft_max"]) | (pb_di_spread_mean >= t["pb_di_spread_mean_soft_min"])

    di_spread = safe_numeric(df.get("trigger_di_spread", pd.Series(np.nan, index=idx)))
    adx = safe_numeric(df.get("trigger_adx", pd.Series(np.nan, index=idx)))
    adx20 = safe_numeric(df.get("trigger_adx_above_20_flag", pd.Series(np.nan, index=idx))) == 1
    vol5 = safe_numeric(df.get("trigger_vol_z_5", pd.Series(np.nan, index=idx)))
    vol20 = safe_numeric(df.get("trigger_vol_z_20", pd.Series(np.nan, index=idx)))
    pb_depth = safe_numeric(df.get("pb_depth_vs_up", pd.Series(np.nan, index=idx)))
    pb_bars = safe_numeric(df.get("pb_bars_vs_up", pd.Series(np.nan, index=idx)))
    pb_hist_neg = safe_numeric(df.get("pb_hist_neg_area", pd.Series(np.nan, index=idx)))
    pb_vol_ratio = safe_numeric(df.get("pb_vol_z_mean_20_vs_up", pd.Series(np.nan, index=idx)))

    components = {
        "type_D": div_label == "D",
        "type_C": div_label == "C",
        "type_D_or_C": div_label.isin(["D", "C"]),
        "bias_positive": bias == 1,
        "best_supertrend_state": st_dir == -1,
        "pullback_trend_preserved_strict": strict_trend_ok,
        "pullback_trend_preserved_soft": soft_trend_ok,
        "trend_confirmation_loose": adx20 | (di_spread > t["trigger_di_spread_lo"]),
        "di_spread_positive": di_spread > t["trigger_di_spread_md"],
        "strong_adx": adx >= t["trigger_adx_hi"],
        "strong_di_spread": di_spread >= t["trigger_di_spread_hi"],
        "volume_confirmation_mid": (vol5 > 0) | (vol20 > t["trigger_vol_z_20_md"]),
        "volume_confirmation_strict": (vol5 >= t["trigger_vol_z_5_md"]) & (vol20 >= t["trigger_vol_z_20_md"]),
        "pullback_not_too_deep": pb_depth <= t["pb_depth_loose_max"],
        "pullback_depth_ok": pb_depth <= t["pb_depth_mid_max"],
        "shallow_pullback": pb_depth <= t["pb_depth_strict_max"],
        "pullback_duration_ok": pb_bars <= t["pb_bars_mid_max"],
        "short_pullback": pb_bars <= t["pb_bars_strict_max"],
        "pullback_hist_ok": pb_hist_neg <= t["pb_hist_neg_mid_max"],
        "pullback_hist_strong": pb_hist_neg <= t["pb_hist_neg_strict_max"],
        "pullback_volume_ratio_ok": pb_vol_ratio <= t["pb_vol_ratio_mid_max"],
        "pullback_volume_ratio_strong": pb_vol_ratio <= t["pb_vol_ratio_strict_max"],
        # 财务评分条件
        "financial_score_good": safe_numeric(df.get("fin_total_score", pd.Series(np.nan, index=idx))) >= 60,
        "financial_growth_good": safe_numeric(df.get("fin_growth_score", pd.Series(np.nan, index=idx))) >= 60,
        "financial_profit_good": safe_numeric(df.get("fin_profit_score", pd.Series(np.nan, index=idx))) >= 60,
        "financial_quality_good": safe_numeric(df.get("fin_quality_score", pd.Series(np.nan, index=idx))) >= 60,
        "financial_cash_good": safe_numeric(df.get("fin_cash_score", pd.Series(np.nan, index=idx))) >= 60,
        "financial_efficiency_good": safe_numeric(df.get("fin_efficiency_score", pd.Series(np.nan, index=idx))) >= 60,
        "financial_momentum_good": safe_numeric(df.get("fin_momentum_score", pd.Series(np.nan, index=idx))) >= 60,
        "financial_any_good": safe_numeric(df.get("fin_total_score", pd.Series(np.nan, index=idx))) >= 50,
    }
    return components, t


def build_rule_specs(df: pd.DataFrame) -> Dict[str, List[Tuple[str, pd.Series]]]:
    c, _ = build_rule_components(df)
    return {
        "宽松总池": [
            ("type_D_or_C", c["type_D_or_C"]),
            ("pullback_trend_preserved_soft", c["pullback_trend_preserved_soft"]),
            ("trend_confirmation_loose", c["trend_confirmation_loose"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
        ],
        "D宽松_基线": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
        ],
        "D宽松_深度增强": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("pullback_depth_ok", c["pullback_depth_ok"]),
        ],
        "D宽松_时长增强": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("pullback_duration_ok", c["pullback_duration_ok"]),
        ],
        "D宽松_量能增强": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("volume_confirmation_mid", c["volume_confirmation_mid"]),
        ],
        "D宽松_动量增强": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("di_spread_positive", c["di_spread_positive"]),
        ],
        "C宽松_观察池": [
            ("type_C", c["type_C"]),
            ("pullback_trend_preserved_soft", c["pullback_trend_preserved_soft"]),
            ("trend_confirmation_loose", c["trend_confirmation_loose"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
        ],
        # 财务增强版规则
        "D宽松_财务增强": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("financial_score_good", c["financial_score_good"]),
        ],
        "D宽松_财务增长": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("financial_growth_good", c["financial_growth_good"]),
        ],
        "D宽松_财务盈利": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("financial_profit_good", c["financial_profit_good"]),
        ],
        "D宽松_财务质量": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("financial_quality_good", c["financial_quality_good"]),
        ],
        "D宽松_财务现金": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("financial_cash_good", c["financial_cash_good"]),
        ],
        "D宽松_财务综合": [
            ("type_D", c["type_D"]),
            ("pullback_not_too_deep", c["pullback_not_too_deep"]),
            ("financial_any_good", c["financial_any_good"]),
        ],
    }

def apply_rule_specs(df: pd.DataFrame, rule_specs: Dict[str, List[Tuple[str, pd.Series]]]) -> pd.DataFrame:
    out = df.copy()
    for rule_name, steps in rule_specs.items():
        mask = pd.Series(True, index=out.index)
        for _, step_mask in steps:
            mask &= step_mask.fillna(False)
        out[f"rule_{rule_name}"] = mask.astype(int)
    return out


def _calc_basic_metrics(df: pd.DataFrame) -> Dict[str, float]:
    def _winrate(col: str) -> float:
        s = safe_numeric(df[col])
        s = s.dropna()
        return float((s > 0).mean()) if not s.empty else np.nan
    return {
        "ret_3_mean": float(safe_numeric(df.get("ret_3", pd.Series(dtype=float))).mean()),
        "ret_5_mean": float(safe_numeric(df.get("ret_5", pd.Series(dtype=float))).mean()),
        "ret_10_mean": float(safe_numeric(df.get("ret_10", pd.Series(dtype=float))).mean()),
        "ret_20_mean": float(safe_numeric(df.get("ret_20", pd.Series(dtype=float))).mean()),
        "win_rate_3": _winrate("ret_3") if "ret_3" in df.columns else np.nan,
        "win_rate_5": _winrate("ret_5") if "ret_5" in df.columns else np.nan,
        "win_rate_10": _winrate("ret_10") if "ret_10" in df.columns else np.nan,
        "mfe_10_mean": float(safe_numeric(df.get("mfe_10", pd.Series(dtype=float))).mean()),
        "mae_10_mean": float(safe_numeric(df.get("mae_10", pd.Series(dtype=float))).mean()),
        "rr_10_mean": float(safe_numeric(df.get("rr_10", pd.Series(dtype=float))).mean()),
    }


def build_rule_summary(df: pd.DataFrame, rule_specs: Dict[str, List[Tuple[str, pd.Series]]]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for rule_name in rule_specs:
        col = f"rule_{rule_name}"
        sub = df[df[col] == 1].copy()
        row = {"rule_name": rule_name, "n": int(len(sub)), "coverage": float(len(sub) / total) if total else np.nan}
        row.update(_calc_basic_metrics(sub))
        rows.append(row)
    return _round_numeric_frame(pd.DataFrame(rows))


def build_rule_yearly_summary(df: pd.DataFrame, rule_specs: Dict[str, List[Tuple[str, pd.Series]]]) -> pd.DataFrame:
    if df.empty or "entry_dt" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["entry_dt"] = pd.to_datetime(tmp["entry_dt"], errors="coerce")
    tmp["year"] = tmp["entry_dt"].dt.year
    tmp = tmp[tmp["year"].notna()].copy()
    rows = []
    for rule_name in rule_specs:
        col = f"rule_{rule_name}"
        sub = tmp[tmp[col] == 1].copy()
        if sub.empty:
            continue
        for year, grp in sub.groupby("year", dropna=True):
            if pd.isna(year):
                continue
            row = {"rule_name": rule_name, "year": int(year), "n": int(len(grp))}
            row.update(_calc_basic_metrics(grp))
            rows.append(row)
    return _round_numeric_frame(pd.DataFrame(rows))


def build_rule_type_summary(df: pd.DataFrame, rule_specs: Dict[str, List[Tuple[str, pd.Series]]]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["divergence_label"] = _resolve_divergence_label(tmp)
    group_cols_base = [c for c in ["divergence_label", "supertrend_dir", "bias"] if c in tmp.columns]
    if not group_cols_base:
        return pd.DataFrame()
    rows = []
    for rule_name in rule_specs:
        col = f"rule_{rule_name}"
        sub = tmp[tmp[col] == 1].copy()
        if sub.empty:
            continue
        grouped = sub.groupby(group_cols_base, dropna=False)
        for keys, grp in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"rule_name": rule_name}
            row.update(dict(zip(group_cols_base, keys)))
            row["n"] = int(len(grp))
            row.update(_calc_basic_metrics(grp))
            rows.append(row)
    return _round_numeric_frame(pd.DataFrame(rows))


def build_rule_debug_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    _, t = build_rule_components(df)
    out = pd.DataFrame({"threshold": list(t.keys()), "value": list(t.values())})
    return _round_numeric_frame(out)


def build_rule_condition_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    c, _ = build_rule_components(df)
    div_label = _resolve_divergence_label(df)
    scopes = {
        "all": pd.Series(True, index=df.index),
        "D_only": div_label == "D",
        "C_only": div_label == "C",
        "H_only": div_label == "H",
    }
    rows = []
    for scope_name, scope_mask in scopes.items():
        denom = int(scope_mask.sum())
        for cond_name, mask in c.items():
            passed = int((scope_mask & mask.fillna(False)).sum())
            rows.append({
                "scope": scope_name,
                "condition_name": cond_name,
                "base_n": denom,
                "pass_n": passed,
                "pass_ratio": float(passed / denom) if denom else np.nan,
            })
    return _round_numeric_frame(pd.DataFrame(rows))

def build_coverage_tradeoff(rule_summary: pd.DataFrame) -> pd.DataFrame:
    if rule_summary.empty:
        return pd.DataFrame()
    out = rule_summary[[c for c in ["rule_name", "n", "coverage", "ret_5_mean", "ret_10_mean", "win_rate_5", "win_rate_10", "rr_10_mean"] if c in rule_summary.columns]].copy()
    order = {"宽松总池": 1, "D宽松_基线": 2, "D宽松_深度增强": 3, "D宽松_时长增强": 4, "D宽松_量能增强": 5, "D宽松_动量增强": 6, "C宽松_观察池": 7}
    out["rule_order"] = out["rule_name"].map(order)
    out = out.sort_values(["rule_order", "coverage"], ascending=[True, False]).drop(columns=["rule_order"])
    return _round_numeric_frame(out)


def build_failed_filter_breakdown(df: pd.DataFrame, rule_specs: Dict[str, List[Tuple[str, pd.Series]]]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for rule_name, steps in rule_specs.items():
        current = pd.Series(True, index=df.index)
        prev_pass = int(current.sum())
        rows.append({"rule_name": rule_name, "step_no": 0, "filter_name": "start", "remaining_n": prev_pass, "removed_n": 0, "remaining_ratio": float(prev_pass / total) if total else np.nan})
        for i, (filter_name, step_mask) in enumerate(steps, start=1):
            step_mask = step_mask.fillna(False)
            before = current.copy()
            current &= step_mask
            remaining = int(current.sum())
            removed = int((before & (~step_mask)).sum())
            rows.append({"rule_name": rule_name, "step_no": i, "filter_name": filter_name, "remaining_n": remaining, "removed_n": removed, "remaining_ratio": float(remaining / total) if total else np.nan})
    return _round_numeric_frame(pd.DataFrame(rows))


def maybe_export_detail_parquets(base_name: str, out_dir: str, structure_table: pd.DataFrame, event_table: pd.DataFrame, label_table: pd.DataFrame, enabled: bool = False) -> List[str]:
    if not enabled:
        return []
    stem = os.path.splitext(os.path.basename(base_name))[0]
    paths: List[str] = []
    for name, df in [('structure_table_pruned', structure_table), ('event_table_pruned', event_table), ('label_table_pruned', label_table)]:
        if df is None or df.empty:
            continue
        path = os.path.join(out_dir, f'{stem}_{name}.parquet')
        df.to_parquet(path, index=False, compression='zstd')
        paths.append(path)
    return paths


def build_feature_inventory(structure_df: pd.DataFrame, event_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    structure_cols = set(structure_df.columns)
    event_cols = set(event_df.columns)
    label_cols = set(label_df.columns)
    all_cols = list(structure_cols | event_cols | label_cols)
    for col in sorted(all_cols):
        group = "other"
        if col.startswith("up_"):
            group = "up_leg"
        elif col.startswith("pb_"):
            group = "pullback"
        elif col.startswith("trigger_") or col in {"divergence", "divergence_id", "bias", "sentiment_code", "supertrend_dir", "price_pivot_dir", "osc_pivot_dir"}:
            group = "event"
        elif col.startswith("ret_") or col.startswith("mfe_") or col.startswith("mae_") or col.startswith("rr_"):
            group = "label"
        elif "trend" in col:
            group = "trend"
        elif "vol_z" in col:
            group = "volume_z"
        elif col.startswith("L1_") or col.startswith("H1_") or col.startswith("L2_"):
            group = "structure_anchor"
        rows.append({
            "feature": col,
            "group": group,
            "in_structure_table": col in structure_cols,
            "in_event_table": col in event_cols,
            "in_label_table": col in label_cols,
        })
    return pd.DataFrame(rows)


def summarize_dataset(structure_df: pd.DataFrame, event_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    rows = [{
        "metric": "n_symbols",
        "value": int(event_df["symbol"].nunique()) if not event_df.empty else 0,
    }, {
        "metric": "n_structures",
        "value": int(len(structure_df)),
    }, {
        "metric": "n_events",
        "value": int(len(event_df)),
    }]
    if not event_df.empty:
        counts = event_df.groupby("structure_id").size()
        rows += [{"metric": "avg_events_per_structure", "value": float(counts.mean())}]
    if not label_df.empty:
        for w in RET_WINDOWS:
            rows += [
                {"metric": f"ret_{w}_mean", "value": float(label_df[f"ret_{w}"].mean()) if f"ret_{w}" in label_df.columns else np.nan},
                {"metric": f"mfe_{w}_mean", "value": float(label_df[f"mfe_{w}"].mean()) if f"mfe_{w}" in label_df.columns else np.nan},
                {"metric": f"mae_{w}_mean", "value": float(label_df[f"mae_{w}"].mean()) if f"mae_{w}" in label_df.columns else np.nan},
            ]
    return pd.DataFrame(rows)


def build_event_quality_summary(label_df: pd.DataFrame) -> pd.DataFrame:
    if label_df.empty:
        return pd.DataFrame()
    group_cols = [c for c in (["divergence_id"] if "divergence_id" in label_df.columns else ["divergence"]) + ["supertrend_dir", "bias"] if c in label_df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = label_df.groupby(group_cols, dropna=False).agg(
        n=("event_id", "count"),
        ret_3_mean=("ret_3", "mean"),
        ret_5_mean=("ret_5", "mean"),
        ret_10_mean=("ret_10", "mean"),
        mfe_10_mean=("mfe_10", "mean"),
        mae_10_mean=("mae_10", "mean"),
        rr_10_mean=("rr_10", "mean"),
    ).reset_index()
    return agg.sort_values(["n"], ascending=False)


def build_bucket_profiles(label_df: pd.DataFrame, features: Sequence[str], target: str = "ret_10", q: int = 5) -> pd.DataFrame:
    if label_df.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for feat in features:
        if feat not in label_df.columns:
            continue
        s = safe_numeric(label_df[feat])
        y = safe_numeric(label_df[target]) if target in label_df.columns else pd.Series(dtype=float)
        mask = s.notna() & y.notna()
        if mask.sum() < max(20, q * 5):
            continue
        try:
            buckets = pd.qcut(s[mask], q=q, labels=False, duplicates="drop")
        except Exception:
            continue
        tmp = pd.DataFrame({"feature": feat, "bucket": buckets, target: y[mask], "mfe_10": label_df.loc[mask, "mfe_10"] if "mfe_10" in label_df.columns else np.nan, "mae_10": label_df.loc[mask, "mae_10"] if "mae_10" in label_df.columns else np.nan})
        prof = tmp.groupby("bucket", dropna=False).agg(
            n=(target, "count"),
            target_mean=(target, "mean"),
            mfe_10_mean=("mfe_10", "mean"),
            mae_10_mean=("mae_10", "mean"),
        ).reset_index()
        for _, r in prof.iterrows():
            rows.append({
                "feature": feat,
                "bucket": int(r["bucket"]) if pd.notna(r["bucket"]) else np.nan,
                "n": int(r["n"]),
                f"{target}_mean": float(r["target_mean"]) if pd.notna(r["target_mean"]) else np.nan,
                "mfe_10_mean": float(r["mfe_10_mean"]) if pd.notna(r["mfe_10_mean"]) else np.nan,
                "mae_10_mean": float(r["mae_10_mean"]) if pd.notna(r["mae_10_mean"]) else np.nan,
            })
    return pd.DataFrame(rows)


def compress_detail_sheet(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return df
    if kind == "structure":
        keep = [c for c in KEEP_STRUCTURE_FEATURES if c in df.columns]
    elif kind == "event":
        keep = [c for c in KEEP_EVENT_FEATURES if c in df.columns]
    else:
        keep = [c for c in KEEP_LABEL_FEATURES if c in df.columns]
    out = df[keep].copy()
    return _round_numeric_frame(out)



def process_single_stock(code: str, idx: int, total: int, bars: int, freq: str, zigzag_cfg: ZigzagMacdConfig, vol_z_windows: Sequence[int]) -> Optional[pd.DataFrame]:
    kline = load_kline(code, freq, bars)
    if kline is None or len(kline) < 120:
        return None
    fac = compute_base_features(kline, zigzag_cfg=zigzag_cfg, vol_z_windows=vol_z_windows)
    fac["symbol"] = code
    fac["datetime"] = fac.index
    print(f"  [{idx}/{total}] {code}: {len(kline)}bars -> base features ready")
    return fac.reset_index(drop=True)


# =========================
# Pipeline
# =========================
def fetch_data_from_db(n_stocks: int, bars: int, freq: str, seed: int, zigzag_cfg: ZigzagMacdConfig, vol_z_windows: Sequence[int], max_workers: int = 1) -> List[pd.DataFrame]:
    stocks = get_stock_pool(n_stocks, seed)
    print(f"股票池抽取: {len(stocks)}只 (seed={seed})")
    all_dfs: List[pd.DataFrame] = []

    if max_workers <= 1:
        for idx, code in enumerate(stocks, start=1):
            try:
                fac = process_single_stock(code, idx, len(stocks), bars, freq, zigzag_cfg, vol_z_windows)
                if fac is not None:
                    all_dfs.append(fac)
            except Exception as exc:
                print(f"  [{idx}/{len(stocks)}] {code} 计算失败: {exc}")
        return all_dfs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {
            executor.submit(process_single_stock, code, idx, len(stocks), bars, freq, zigzag_cfg, vol_z_windows): (idx, code)
            for idx, code in enumerate(stocks, start=1)
        }
        for future in as_completed(future_to_meta):
            idx, code = future_to_meta[future]
            try:
                fac = future.result()
                if fac is not None:
                    all_dfs.append(fac)
            except Exception as exc:
                print(f"  [{idx}/{len(stocks)}] {code} 计算失败: {exc}")
    return all_dfs


def _process_single_stock(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    处理单只股票的辅助函数（模块级别定义，用于多进程pickle序列化）
    
    Returns:
        (structure_df, event_df, label_df) 或 None（如果处理失败或无事件）
    """
    if df.empty:
        return None
    try:
        symbol = str(df["symbol"].iloc[0])
        work = df.copy()
        work.index = pd.to_datetime(work["datetime"])
        structure_df, event_df = build_structures_and_events(work, symbol)
        if event_df.empty:
            return None
        label_df = add_event_labels(event_df, work)
        return (structure_df, event_df, label_df)
    except Exception:
        return None


def prepare_dataset(
    all_dfs: List[pd.DataFrame],
    use_parallel: bool = True,
    n_workers: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    提取结构与事件，支持并行处理
    
    Args:
        all_dfs: 每只股票的DataFrame列表
        use_parallel: 是否使用并行处理（默认True）
        n_workers: 并行进程数，默认使用CPU核心数（最多8个）
    
    Returns:
        (structure_table, event_table, label_table)
    """
    structure_frames: List[pd.DataFrame] = []
    event_frames: List[pd.DataFrame] = []
    label_frames: List[pd.DataFrame] = []
    
    if use_parallel and len(all_dfs) > 10:
        # 并行处理模式
        if n_workers is None:
            n_workers = min(cpu_count(), 8)
        
        print(f"[2/3] 提取结构与事件（并行模式，{n_workers}个进程）...")
        
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_single_stock, all_dfs, chunksize=max(1, len(all_dfs) // n_workers // 4)),
                total=len(all_dfs),
                desc="处理进度",
                position=0, leave=True
            ))
        
        # 收集结果
        for result in results:
            if result is not None:
                s, e, l = result
                structure_frames.append(s)
                event_frames.append(e)
                label_frames.append(l)
    else:
        # 串行处理模式（股票数较少时）
        for df in tqdm(all_dfs, desc="[2/3] 提取结构与事件", position=0, leave=True):
            result = _process_single_stock(df)
            if result is not None:
                s, e, l = result
                structure_frames.append(s)
                event_frames.append(e)
                label_frames.append(l)

    structure_table = pd.concat(structure_frames, ignore_index=True) if structure_frames else pd.DataFrame()
    event_table = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    label_table = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    return structure_table, event_table, label_table





# =========================
# GBDT auxiliary experiment (D宽松样本)
# =========================
GBDT_FEATURES = [
    "trigger_vol_z_5", "trigger_vol_z_20", "trigger_adx", "trigger_di_spread",
    "trigger_hist", "trigger_ret_1d",
    "pb_depth_vs_up", "pb_bars_vs_up", "pb_hist_mean",
    "pb_adx_mean", "pb_di_spread_mean", "pb_vol_z_mean_20_vs_up",
    "up_return", "up_hist_mean",
    # 财务评分因子
    "fin_total_score", "fin_growth_score", "fin_profit_score",
    "fin_quality_score", "fin_cash_score", "fin_efficiency_score", "fin_momentum_score",
]


def _require_sklearn() -> None:
    if GradientBoostingClassifier is None or GradientBoostingRegressor is None or SimpleImputer is None:
        raise RuntimeError(
            "缺少 scikit-learn。请先安装: pip install scikit-learn"
        )


def build_gbdt_sample(df: pd.DataFrame, rule_col: str = "rule_D宽松_基线") -> pd.DataFrame:
    if df.empty or rule_col not in df.columns:
        return pd.DataFrame()
    out = df[df[rule_col] == 1].copy()
    if out.empty:
        return out
    out["entry_dt"] = pd.to_datetime(out.get("entry_dt"), errors="coerce")
    out["year"] = out["entry_dt"].dt.year
    out["y_cls_5"] = (safe_numeric(out["ret_5"]) > 0).astype(float)
    if "high_quality_flag_short" in out.columns:
        out["y_cls_hq"] = safe_numeric(out["high_quality_flag_short"]).fillna(0).astype(float)
    else:
        # 兜底：用5/10日收益与回撤的简单规则构造质量标签，避免旧结果表缺字段时报错
        out["y_cls_hq"] = ((safe_numeric(out["ret_5"]) > 0) & (safe_numeric(out.get("rr_10", pd.Series(index=out.index, dtype=float))).fillna(0) > 0)).astype(float)
    out["y_reg_10"] = safe_numeric(out["ret_10"])
    keep = [c for c in ["symbol", "event_id", "structure_id", "entry_dt", "year"] + GBDT_FEATURES + ["y_cls_5", "y_cls_hq", "y_reg_10", "ret_5", "ret_10"] if c in out.columns]
    return out[keep].copy()


def build_gbdt_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = [{
        "scope": "overall",
        "n": int(len(df)),
        "pos_rate_cls_5": float(safe_numeric(df["y_cls_5"]).mean()),
        "pos_rate_cls_hq": float(safe_numeric(df["y_cls_hq"]).mean()),
        "ret_5_mean": float(safe_numeric(df["ret_5"]).mean()),
        "ret_10_mean": float(safe_numeric(df["ret_10"]).mean()),
    }]
    if "year" in df.columns and df["year"].notna().any():
        for year, grp in df.groupby("year", dropna=True):
            if pd.isna(year):
                continue
            rows.append({
                "scope": f"year_{int(year)}",
                "year": int(year),
                "n": int(len(grp)),
                "pos_rate_cls_5": float(safe_numeric(grp["y_cls_5"]).mean()),
                "pos_rate_cls_hq": float(safe_numeric(grp["y_cls_hq"]).mean()),
                "ret_5_mean": float(safe_numeric(grp["ret_5"]).mean()),
                "ret_10_mean": float(safe_numeric(grp["ret_10"]).mean()),
            })
    return _round_numeric_frame(pd.DataFrame(rows))


def _prepare_xy(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: Sequence[str], target_col: str):
    _require_sklearn()
    feat_cols = [c for c in feature_cols if c in train_df.columns and c in valid_df.columns]
    xtr = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    xva = valid_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    ytr = pd.to_numeric(train_df[target_col], errors="coerce")
    yva = pd.to_numeric(valid_df[target_col], errors="coerce")
    mask_tr = ytr.notna()
    mask_va = yva.notna()
    xtr = xtr.loc[mask_tr]
    xva = xva.loc[mask_va]
    ytr = ytr.loc[mask_tr]
    yva = yva.loc[mask_va]
    imp = SimpleImputer(strategy="median")
    xtr_imp = imp.fit_transform(xtr)
    xva_imp = imp.transform(xva)
    return feat_cols, xtr_imp, xva_imp, ytr.to_numpy(), yva.to_numpy()


def _build_year_splits(df: pd.DataFrame) -> List[Tuple[List[int], int]]:
    years = sorted(int(y) for y in pd.Series(df.get("year")).dropna().unique())
    splits: List[Tuple[List[int], int]] = []
    for val_year in years:
        train_years = [y for y in years if y < val_year]
        if len(train_years) < 2:
            continue
        splits.append((train_years, val_year))
    return splits


def _fit_gbdt_classifier(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: Sequence[str], target_col: str):
    feat_cols, xtr, xva, ytr, yva = _prepare_xy(train_df, valid_df, feature_cols, target_col)
    if len(np.unique(ytr)) < 2 or len(yva) == 0:
        return None
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(xtr, ytr)
    prob = model.predict_proba(xva)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = np.nan
    if len(np.unique(yva)) >= 2 and roc_auc_score is not None:
        auc = float(roc_auc_score(yva, prob))
    acc = float(accuracy_score(yva, pred)) if accuracy_score is not None else np.nan
    return feat_cols, model, prob, yva, acc, auc


def _fit_gbdt_regressor(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: Sequence[str], target_col: str):
    feat_cols, xtr, xva, ytr, yva = _prepare_xy(train_df, valid_df, feature_cols, target_col)
    if len(ytr) < 50 or len(yva) == 0:
        return None
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(xtr, ytr)
    pred = model.predict(xva)
    return feat_cols, model, pred, yva


def _feature_importance_frame(feature_cols: Sequence[str], importances: Sequence[float], model_name: str, split_name: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "feature": list(feature_cols),
        "importance": list(importances),
        "model_name": model_name,
        "split_name": split_name,
    }).sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return _round_numeric_frame(out)


def build_gbdt_feature_importance_tables(df: pd.DataFrame, feature_cols: Sequence[str]):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    splits = _build_year_splits(df)
    fi_cls5_parts: List[pd.DataFrame] = []
    fi_hq_parts: List[pd.DataFrame] = []
    fi_reg10_parts: List[pd.DataFrame] = []
    val_rows: List[Dict[str, float]] = []

    # 全样本重要性
    whole = df.copy()
    if len(whole) >= 100:
        try:
            imp = SimpleImputer(strategy="median")
            X = imp.fit_transform(whole[list(feature_cols)].apply(pd.to_numeric, errors="coerce"))
            y1 = pd.to_numeric(whole["y_cls_5"], errors="coerce").fillna(0).astype(int)
            if len(np.unique(y1)) >= 2:
                m1 = GradientBoostingClassifier(random_state=RANDOM_STATE)
                m1.fit(X, y1)
                fi_cls5_parts.append(_feature_importance_frame(feature_cols, m1.feature_importances_, "cls_ret5", "all_in_sample"))
            y2 = pd.to_numeric(whole["y_cls_hq"], errors="coerce").fillna(0).astype(int)
            if len(np.unique(y2)) >= 2:
                m2 = GradientBoostingClassifier(random_state=RANDOM_STATE)
                m2.fit(X, y2)
                fi_hq_parts.append(_feature_importance_frame(feature_cols, m2.feature_importances_, "cls_hq", "all_in_sample"))
            y3 = pd.to_numeric(whole["y_reg_10"], errors="coerce")
            mask3 = y3.notna()
            if int(mask3.sum()) >= 100:
                mr = GradientBoostingRegressor(random_state=RANDOM_STATE)
                mr.fit(X[mask3.values], y3.loc[mask3].to_numpy())
                fi_reg10_parts.append(_feature_importance_frame(feature_cols, mr.feature_importances_, "reg_ret10", "all_in_sample"))
        except Exception:
            pass

    for train_years, val_year in splits:
        train_df = df[df["year"].isin(train_years)].copy()
        valid_df = df[df["year"] == val_year].copy()
        split_name = f"train_{min(train_years)}_{max(train_years)}_val_{val_year}"

        cls5 = _fit_gbdt_classifier(train_df, valid_df, feature_cols, "y_cls_5")
        if cls5 is not None:
            feat_cols, model, prob, yva, acc, auc = cls5
            fi_cls5_parts.append(_feature_importance_frame(feat_cols, model.feature_importances_, "cls_ret5", split_name))
            valid_eval = valid_df.loc[pd.to_numeric(valid_df["y_cls_5"], errors="coerce").notna()].copy()
            valid_eval = valid_eval.iloc[:len(prob)].copy()
            valid_eval["score"] = prob
            top_n = max(1, int(len(valid_eval) * 0.2))
            top = valid_eval.sort_values("score", ascending=False).head(top_n)
            val_rows.append({
                "task": "cls_ret5", "split_name": split_name, "train_years": ",".join(map(str, train_years)),
                "val_year": int(val_year), "train_n": int(len(train_df)), "valid_n": int(len(valid_df)),
                "accuracy": acc, "auc": auc,
                "top20_ret_5_mean": float(safe_numeric(top["ret_5"]).mean()),
                "top20_ret_10_mean": float(safe_numeric(top["ret_10"]).mean()),
            })

        cls_hq = _fit_gbdt_classifier(train_df, valid_df, feature_cols, "y_cls_hq")
        if cls_hq is not None:
            feat_cols, model, prob, yva, acc, auc = cls_hq
            fi_hq_parts.append(_feature_importance_frame(feat_cols, model.feature_importances_, "cls_hq", split_name))
            valid_eval = valid_df.loc[pd.to_numeric(valid_df["y_cls_hq"], errors="coerce").notna()].copy()
            valid_eval = valid_eval.iloc[:len(prob)].copy()
            valid_eval["score"] = prob
            top_n = max(1, int(len(valid_eval) * 0.2))
            top = valid_eval.sort_values("score", ascending=False).head(top_n)
            val_rows.append({
                "task": "cls_hq", "split_name": split_name, "train_years": ",".join(map(str, train_years)),
                "val_year": int(val_year), "train_n": int(len(train_df)), "valid_n": int(len(valid_df)),
                "accuracy": acc, "auc": auc,
                "top20_ret_5_mean": float(safe_numeric(top["ret_5"]).mean()),
                "top20_ret_10_mean": float(safe_numeric(top["ret_10"]).mean()),
            })

        reg10 = _fit_gbdt_regressor(train_df, valid_df, feature_cols, "y_reg_10")
        if reg10 is not None:
            feat_cols, model, pred, yva = reg10
            fi_reg10_parts.append(_feature_importance_frame(feat_cols, model.feature_importances_, "reg_ret10", split_name))
            valid_eval = valid_df.loc[pd.to_numeric(valid_df["y_reg_10"], errors="coerce").notna()].copy()
            valid_eval = valid_eval.iloc[:len(pred)].copy()
            valid_eval["score"] = pred
            top_n = max(1, int(len(valid_eval) * 0.2))
            top = valid_eval.sort_values("score", ascending=False).head(top_n)
            val_rows.append({
                "task": "reg_ret10", "split_name": split_name, "train_years": ",".join(map(str, train_years)),
                "val_year": int(val_year), "train_n": int(len(train_df)), "valid_n": int(len(valid_df)),
                "accuracy": np.nan, "auc": np.nan,
                "top20_ret_5_mean": float(safe_numeric(top["ret_5"]).mean()),
                "top20_ret_10_mean": float(safe_numeric(top["ret_10"]).mean()),
            })

    fi_cls5 = pd.concat(fi_cls5_parts, ignore_index=True) if fi_cls5_parts else pd.DataFrame()
    fi_hq = pd.concat(fi_hq_parts, ignore_index=True) if fi_hq_parts else pd.DataFrame()
    fi_reg10 = pd.concat(fi_reg10_parts, ignore_index=True) if fi_reg10_parts else pd.DataFrame()
    yearly_validation = _round_numeric_frame(pd.DataFrame(val_rows))

    return fi_cls5, fi_hq, fi_reg10, yearly_validation, build_gbdt_rule_hint_summary(fi_cls5, fi_hq, fi_reg10)


def build_gbdt_rule_hint_summary(fi_cls5: pd.DataFrame, fi_hq: pd.DataFrame, fi_reg10: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tag, df in [("cls_ret5", fi_cls5), ("cls_hq", fi_hq), ("reg_ret10", fi_reg10)]:
        if df is None or df.empty:
            continue
        agg = df.groupby("feature", dropna=False).agg(
            mean_importance=("importance", "mean"),
            best_rank=("rank", "min"),
            avg_rank=("rank", "mean"),
            n_splits=("split_name", "nunique"),
        ).reset_index()
        agg["task"] = tag
        parts.append(agg)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    out = merged.groupby("feature", dropna=False).agg(
        mean_importance=("mean_importance", "mean"),
        best_rank=("best_rank", "min"),
        avg_rank=("avg_rank", "mean"),
        tasks_supported=("task", "nunique"),
        n_splits=("n_splits", "max"),
    ).reset_index()

    def _recommend(row: pd.Series) -> str:
        if row["tasks_supported"] >= 2 and row["avg_rank"] <= 6:
            return "硬过滤候选"
        if row["tasks_supported"] >= 2 and row["avg_rank"] <= 12:
            return "软排序候选"
        if row["best_rank"] <= 8:
            return "观察项"
        return "可降级/删除"

    out["recommendation"] = out.apply(_recommend, axis=1)
    out = out.sort_values(["tasks_supported", "avg_rank", "mean_importance"], ascending=[False, True, False]).reset_index(drop=True)
    return _round_numeric_frame(out)

def main() -> None:
    parser = argparse.ArgumentParser(description="日线底背离买点质量研究脚本")
    parser.add_argument("--n-stocks", type=int, default=100)
    parser.add_argument("--bars", type=int, default=800)
    parser.add_argument("--freq", default="d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="data_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--excel-name", type=str, default="daily_bottom_div_quality_report.xlsx")
    parser.add_argument("--zigzag-length", type=int, default=5)
    parser.add_argument("--supertrend-history", type=int, default=4)
    parser.add_argument("--atr-periods", type=int, default=22)
    parser.add_argument("--atr-mult", type=float, default=1.0)
    parser.add_argument("--osc-short", type=int, default=12)
    parser.add_argument("--osc-long", type=int, default=26)
    parser.add_argument("--osc-len", type=int, default=22)
    parser.add_argument("--vol-z-windows", type=str, default="5,10,20,40")
    parser.add_argument("--max-workers", type=int, default=1, help="K线加载并行线程数，1表示关闭多线程")
    parser.add_argument("--prepare-workers", type=int, default=None, help="结构提取并行进程数，默认自动（最多8）")
    parser.add_argument("--no-parallel-prepare", action="store_true", help="禁用结构提取并行处理")
    parser.add_argument("--excel-detail", type=str, default="stats_only", choices=["stats_only", "compact", "full"], help="Excel输出级别，默认仅统计结果")
    parser.add_argument("--export-detail-parquet", action="store_true", help="额外导出裁剪后的明细parquet，默认关闭")
    args = parser.parse_args()

    if str(args.freq).lower() != "d":
        raise ValueError("当前脚本固定只做日线，请使用 --freq d")

    vol_z_windows = tuple(int(x) for x in str(args.vol_z_windows).split(",") if str(x).strip())
    zigzag_cfg = ZigzagMacdConfig(
        zigzag_length=args.zigzag_length,
        supertrend_history=args.supertrend_history,
        atr_periods=args.atr_periods,
        atr_mult=args.atr_mult,
        oscillator_length=args.osc_len,
        oscillator_short_length=args.osc_short,
        oscillator_long_length=args.osc_long,
    )

    print("研究目标: 日线底背离买点质量研究（保留同一结构内全部底背离样本）")
    cache_path = get_cache_path(args.cache_dir, args.n_stocks, args.bars, args.freq, args.seed)

    all_dfs: List[pd.DataFrame] = []
    if not args.no_cache and not args.refresh_cache and cache_exists(cache_path):
        print(f"\n发现缓存文件，正在加载: {cache_path.replace('.pkl', '.parquet')}")
        cached = load_cache(cache_path)
        if cached is not None:
            all_dfs = cached.get("all_dfs", [])
            metadata = cached.get("metadata", {})
            print(f"缓存加载成功: {len(all_dfs)}只股票, 创建时间: {metadata.get('created_at', 'unknown')}")

    if not all_dfs:
        print("\n从数据库获取数据并计算底座特征...")
        all_dfs = fetch_data_from_db(
            args.n_stocks, args.bars, args.freq, args.seed,
            zigzag_cfg, vol_z_windows, max_workers=max(1, int(args.max_workers))
        )
        if not args.no_cache and all_dfs:
            metadata = {
                "n_stocks": args.n_stocks,
                "bars": args.bars,
                "freq": args.freq,
                "seed": args.seed,
                "created_at": str(pd.Timestamp.now()),
                "stock_count": len(all_dfs),
            }
            save_cache({"all_dfs": all_dfs, "metadata": metadata}, cache_path)
            print(f"缓存已保存: {cache_path.replace('.pkl', '.parquet')}")

    structure_table, event_table, label_table = prepare_dataset(
        all_dfs,
        use_parallel=not args.no_parallel_prepare,
        n_workers=args.prepare_workers,
    )

    # 添加财务评分因子
    print("\n添加财务评分因子...")
    event_table = add_financial_score_factors(event_table)
    print(f"财务评分因子已添加，字段: {[c for c in event_table.columns if c.startswith('fin_')]}")

    analysis_table = build_analysis_table(structure_table, event_table, label_table)
    dataset_summary = summarize_dataset(structure_table, event_table, analysis_table)
    event_quality_summary = build_event_quality_summary(analysis_table)
    yearly_summary = build_yearly_summary(analysis_table)
    type_year_summary = build_type_year_summary(analysis_table)
    bucket_features = [
        feat for feat in DEFAULT_BUCKET_FEATURES
        if feat in CORE_FEATURES_FOR_SUMMARY and feat in analysis_table.columns
    ]
    bucket_profiles = build_bucket_profiles(analysis_table, bucket_features, target="ret_10", q=5)
    feature_keep_drop_report = build_feature_keep_drop_report(structure_table, event_table, analysis_table)
    feature_inventory = build_feature_inventory(structure_table, event_table, analysis_table)
    rule_specs = build_rule_specs(analysis_table)
    analysis_table = apply_rule_specs(analysis_table, rule_specs)
    rule_summary = build_rule_summary(analysis_table, rule_specs)
    rule_yearly_summary = build_rule_yearly_summary(analysis_table, rule_specs)
    rule_type_summary = build_rule_type_summary(analysis_table, rule_specs)
    coverage_tradeoff = build_coverage_tradeoff(rule_summary)
    failed_filter_breakdown = build_failed_filter_breakdown(analysis_table, rule_specs)
    rule_debug_thresholds = build_rule_debug_thresholds(analysis_table)
    rule_condition_diagnostics = build_rule_condition_diagnostics(analysis_table)

    gbdt_sample = build_gbdt_sample(analysis_table, rule_col="rule_D宽松_基线")
    gbdt_dataset_summary = build_gbdt_dataset_summary(gbdt_sample)
    gbdt_feature_importance_cls_5 = pd.DataFrame()
    gbdt_feature_importance_cls_hq = pd.DataFrame()
    gbdt_feature_importance_reg_10 = pd.DataFrame()
    gbdt_yearly_validation = pd.DataFrame()
    gbdt_rule_hint_summary = pd.DataFrame()
    if not gbdt_sample.empty:
        try:
            (
                gbdt_feature_importance_cls_5,
                gbdt_feature_importance_cls_hq,
                gbdt_feature_importance_reg_10,
                gbdt_yearly_validation,
                gbdt_rule_hint_summary,
            ) = build_gbdt_feature_importance_tables(gbdt_sample, [f for f in GBDT_FEATURES if f in gbdt_sample.columns])
        except Exception as exc:
            gbdt_dataset_summary = pd.concat([
                gbdt_dataset_summary,
                pd.DataFrame([{"scope": "error", "message": str(exc)}])
            ], ignore_index=True)

    result_tables: Dict[str, pd.DataFrame] = {
        "dataset_summary": dataset_summary,
        "event_quality_summary": event_quality_summary,
        "yearly_summary": yearly_summary,
        "type_year_summary": type_year_summary,
        "bucket_profiles": bucket_profiles,
        "rule_summary": rule_summary,
        "rule_yearly_summary": rule_yearly_summary,
        "rule_type_summary": rule_type_summary,
        "coverage_tradeoff": coverage_tradeoff,
        "failed_filter_breakdown": failed_filter_breakdown,
        "rule_debug_thresholds": rule_debug_thresholds,
        "rule_condition_diagnostics": rule_condition_diagnostics,
        "gbdt_dataset_summary": gbdt_dataset_summary,
        "gbdt_feature_importance_cls_5": gbdt_feature_importance_cls_5,
        "gbdt_feature_importance_cls_hq": gbdt_feature_importance_cls_hq,
        "gbdt_feature_importance_reg_10": gbdt_feature_importance_reg_10,
        "gbdt_yearly_validation": gbdt_yearly_validation,
        "gbdt_rule_hint_summary": gbdt_rule_hint_summary,
        "feature_keep_drop_report": feature_keep_drop_report,
        "feature_inventory": feature_inventory,
    }

    if args.excel_detail in {"compact", "full"}:
        structure_for_excel = structure_table if args.excel_detail == "full" else compress_detail_sheet(structure_table, "structure")
        event_for_excel = event_table if args.excel_detail == "full" else compress_detail_sheet(event_table, "event")
        label_for_excel = analysis_table if args.excel_detail == "full" else compress_detail_sheet(analysis_table, "label")
        result_tables = {
            "dataset_summary": dataset_summary,
            "structure_table": structure_for_excel,
            "event_table": event_for_excel,
            "label_table": label_for_excel,
            "event_quality_summary": event_quality_summary,
            "yearly_summary": yearly_summary,
            "type_year_summary": type_year_summary,
            "bucket_profiles": bucket_profiles,
            "rule_summary": rule_summary,
            "rule_yearly_summary": rule_yearly_summary,
            "rule_type_summary": rule_type_summary,
            "coverage_tradeoff": coverage_tradeoff,
            "failed_filter_breakdown": failed_filter_breakdown,
            "rule_debug_thresholds": rule_debug_thresholds,
            "rule_condition_diagnostics": rule_condition_diagnostics,
            "gbdt_dataset_summary": gbdt_dataset_summary,
            "gbdt_feature_importance_cls_5": gbdt_feature_importance_cls_5,
            "gbdt_feature_importance_cls_hq": gbdt_feature_importance_cls_hq,
            "gbdt_feature_importance_reg_10": gbdt_feature_importance_reg_10,
            "gbdt_yearly_validation": gbdt_yearly_validation,
            "gbdt_rule_hint_summary": gbdt_rule_hint_summary,
            "feature_keep_drop_report": feature_keep_drop_report,
            "feature_inventory": feature_inventory,
        }

    workbook_path = export_results_workbook(result_tables, OUT_DIR, args.excel_name)
    parquet_paths = maybe_export_detail_parquets(
        args.excel_name,
        OUT_DIR,
        structure_table,
        event_table,
        analysis_table,
        enabled=bool(args.export_detail_parquet),
    )
    print(f"\n完成，Excel 已导出: {workbook_path}")
    if parquet_paths:
        print("裁剪明细 Parquet:")
        for p in parquet_paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
