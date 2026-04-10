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

OUT_DIR = "daily_bottom_div_quality_output"
os.makedirs(OUT_DIR, exist_ok=True)

RET_WINDOWS = [3, 5, 10, 20]
ENTRY_LAG = 1
ENTRY_PRICE_MODE = "next_open"
DEFAULT_VOL_Z_WINDOWS = (5, 10, 20, 40)
RANDOM_STATE = 42


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
# Utilities
# =========================
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def rr_from_ret_dd(ret: float, dd: float) -> float:
    if pd.isna(ret) or pd.isna(dd) or dd == 0:
        return np.nan
    return float(ret / abs(dd))


def clip_value(x: float, lower: float, upper: float) -> float:
    if pd.isna(x):
        return np.nan
    return float(np.clip(float(x), lower, upper))


def rank_pct_series(s: pd.Series) -> pd.Series:
    s = safe_numeric(s)
    mask = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if mask.sum() == 0:
        return out
    ranked = s[mask].rank(method='average', pct=True) * 100.0
    out.loc[mask] = ranked.astype(float)
    return out


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
                "README / dataset_summary / structure_table / event_table / label_table / event_quality_summary / bucket_profiles / feature_inventory",
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

    # === 3-5天短持仓质量评分器（服务后续GBDT标签工程） ===
    ret_3 = safe_numeric(out.get("ret_3", pd.Series(index=out.index, dtype=float)))
    ret_5 = safe_numeric(out.get("ret_5", pd.Series(index=out.index, dtype=float)))
    mae_3 = safe_numeric(out.get("mae_3", pd.Series(index=out.index, dtype=float)))
    mae_5 = safe_numeric(out.get("mae_5", pd.Series(index=out.index, dtype=float)))
    rr_3 = safe_numeric(out.get("rr_3", pd.Series(index=out.index, dtype=float)))
    rr_5 = safe_numeric(out.get("rr_5", pd.Series(index=out.index, dtype=float)))

    out["reward_ret_raw_short"] = (
        ret_3.apply(lambda x: clip_value(x, -0.12, 0.12)) * 0.40
        + ret_5.apply(lambda x: clip_value(x, -0.18, 0.18)) * 0.60
    )
    out["penalty_dd_raw_short"] = (
        mae_3.fillna(0.0).apply(lambda x: abs(min(float(x), 0.0))) * 0.40
        + mae_5.fillna(0.0).apply(lambda x: abs(min(float(x), 0.0))) * 0.60
    )
    out["reward_rr_raw_short"] = (
        rr_3.apply(lambda x: clip_value(x, -3.0, 3.0)) * 0.40
        + rr_5.apply(lambda x: clip_value(x, -4.0, 4.0)) * 0.60
    )
    out["quality_raw_short"] = (
        out["reward_ret_raw_short"] * 0.35
        + out["reward_rr_raw_short"] * 0.15
        - out["penalty_dd_raw_short"] * 0.50
    )

    # 短持仓回撤硬惩罚
    hard_penalty = pd.Series(0.0, index=out.index, dtype=float)
    hard_penalty += np.where(mae_3 <= -0.05, 0.03, 0.0)
    hard_penalty += np.where(mae_5 <= -0.08, 0.05, 0.0)
    hard_penalty += np.where(mae_5 <= -0.10, 0.08, 0.0)
    out["quality_hard_penalty_short"] = hard_penalty
    out["quality_raw_short"] = out["quality_raw_short"] - out["quality_hard_penalty_short"]
    out["quality_score_short"] = rank_pct_series(out["quality_raw_short"])
    out["quality_bucket_short"] = pd.cut(
        out["quality_score_short"],
        bins=[-np.inf, 40, 80, np.inf],
        labels=["low", "mid", "high"],
    ).astype(object)
    out["high_quality_flag_short"] = (
        (ret_5 > 0)
        & (mae_5 > -0.06)
        & (rr_5 > 0.5)
    ).astype(float)

    return out



# =========================
# Summary / buckets
# =========================
def build_feature_inventory(structure_df: pd.DataFrame, event_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    structure_cols = set(structure_df.columns)
    event_cols = set(event_df.columns)
    label_cols = set(label_df.columns)
    all_cols = list(structure_cols | event_cols | label_cols)
    id_like = {"symbol", "structure_id", "event_id"}
    time_like = {"L1_dt", "H1_dt", "L2_dt", "trigger_dt", "entry_dt"}
    label_prefixes = ("ret_", "mfe_", "mae_", "rr_")
    derived_label_cols = {
        "reward_ret_raw_short", "penalty_dd_raw_short", "reward_rr_raw_short",
        "quality_hard_penalty_short", "quality_raw_short", "quality_score_short",
        "quality_bucket_short", "high_quality_flag_short",
    }
    for col in sorted(all_cols):
        group = "other"
        if col.startswith("up_"):
            group = "up_leg"
        elif col.startswith("pb_"):
            group = "pullback"
        elif col.startswith("trigger_") or col in {"divergence", "divergence_id", "bias", "sentiment_code", "supertrend_dir", "price_pivot_dir", "osc_pivot_dir"}:
            group = "event"
        elif col.startswith(label_prefixes) or col in derived_label_cols:
            group = "label"
        elif "trend" in col:
            group = "trend"
        elif "vol_z" in col:
            group = "volume_z"
        elif col.startswith("L1_") or col.startswith("H1_") or col.startswith("L2_"):
            group = "structure_anchor"
        is_label = col.startswith(label_prefixes) or col in derived_label_cols
        is_id = col in id_like
        is_time_key = col in time_like or col.endswith("_year") or col.endswith("_month") or col.endswith("_ym") or col.endswith("_weekday")
        is_train_feature = (not is_label) and (not is_id) and (not is_time_key)
        rows.append({
            "feature": col,
            "group": group,
            "in_structure_table": col in structure_cols,
            "in_event_table": col in event_cols,
            "in_label_table": col in label_cols,
            "is_label": is_label,
            "is_id": is_id,
            "is_time_key": is_time_key,
            "is_train_feature_candidate": is_train_feature,
        })
    return pd.DataFrame(rows)


def build_model_feature_manifest(structure_df: pd.DataFrame, event_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    inv = build_feature_inventory(structure_df, event_df, label_df).copy()
    inv["source_table"] = np.select(
        [inv["in_event_table"], inv["in_structure_table"], inv["in_label_table"]],
        ["event_table", "structure_table", "label_table"],
        default="other",
    )
    inv["notes"] = ""
    inv.loc[inv["is_label"], "notes"] = "future label / derived label, 禁止作为训练特征"
    inv.loc[inv["is_id"], "notes"] = "ID键，仅用于关联或分组"
    inv.loc[inv["is_time_key"], "notes"] = "时间键，用于切分与分析，不直接喂模型"
    return inv[[
        "feature", "group", "source_table", "in_structure_table", "in_event_table", "in_label_table",
        "is_label", "is_id", "is_time_key", "is_train_feature_candidate", "notes"
    ]].sort_values(["is_train_feature_candidate", "group", "feature"], ascending=[False, True, True]).reset_index(drop=True)


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
                {"metric": f"rr_{w}_mean", "value": float(label_df[f"rr_{w}"].mean()) if f"rr_{w}" in label_df.columns else np.nan},
            ]
        rows += [
            {"metric": "quality_score_short_mean", "value": float(label_df["quality_score_short"].mean()) if "quality_score_short" in label_df.columns else np.nan},
            {"metric": "quality_score_short_p80", "value": float(label_df["quality_score_short"].quantile(0.8)) if "quality_score_short" in label_df.columns else np.nan},
            {"metric": "high_quality_flag_short_mean", "value": float(label_df["high_quality_flag_short"].mean()) if "high_quality_flag_short" in label_df.columns else np.nan},
        ]
    return pd.DataFrame(rows)


def build_event_quality_summary(label_df: pd.DataFrame) -> pd.DataFrame:
    if label_df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["divergence_id", "supertrend_dir", "bias"] if c in label_df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = label_df.groupby(group_cols, dropna=False).agg(
        n=("event_id", "count"),
        quality_score_short_mean=("quality_score_short", "mean"),
        high_quality_flag_short_mean=("high_quality_flag_short", "mean"),
        ret_3_mean=("ret_3", "mean"),
        ret_5_mean=("ret_5", "mean"),
        mae_3_mean=("mae_3", "mean"),
        mae_5_mean=("mae_5", "mean"),
        rr_3_mean=("rr_3", "mean"),
        rr_5_mean=("rr_5", "mean"),
    ).reset_index()
    return agg.sort_values(["quality_score_short_mean", "n"], ascending=[False, False])


def build_bucket_profiles(label_df: pd.DataFrame, features: Sequence[str], target: str = "quality_score_short", q: int = 5) -> pd.DataFrame:
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
        tmp = pd.DataFrame({
            "feature": feat,
            "bucket": buckets,
            target: y[mask],
            "ret_3": label_df.loc[mask, "ret_3"] if "ret_3" in label_df.columns else np.nan,
            "ret_5": label_df.loc[mask, "ret_5"] if "ret_5" in label_df.columns else np.nan,
            "mae_3": label_df.loc[mask, "mae_3"] if "mae_3" in label_df.columns else np.nan,
            "mae_5": label_df.loc[mask, "mae_5"] if "mae_5" in label_df.columns else np.nan,
            "high_quality_flag_short": label_df.loc[mask, "high_quality_flag_short"] if "high_quality_flag_short" in label_df.columns else np.nan,
        })
        prof = tmp.groupby("bucket", dropna=False).agg(
            n=(target, "count"),
            target_mean=(target, "mean"),
            ret_3_mean=("ret_3", "mean"),
            ret_5_mean=("ret_5", "mean"),
            mae_3_mean=("mae_3", "mean"),
            mae_5_mean=("mae_5", "mean"),
            high_quality_flag_short_mean=("high_quality_flag_short", "mean"),
        ).reset_index()
        for _, r in prof.iterrows():
            rows.append({
                "feature": feat,
                "bucket": int(r["bucket"]) if pd.notna(r["bucket"]) else np.nan,
                "n": int(r["n"]),
                f"{target}_mean": float(r["target_mean"]) if pd.notna(r["target_mean"]) else np.nan,
                "ret_3_mean": float(r["ret_3_mean"]) if pd.notna(r["ret_3_mean"]) else np.nan,
                "ret_5_mean": float(r["ret_5_mean"]) if pd.notna(r["ret_5_mean"]) else np.nan,
                "mae_3_mean": float(r["mae_3_mean"]) if pd.notna(r["mae_3_mean"]) else np.nan,
                "mae_5_mean": float(r["mae_5_mean"]) if pd.notna(r["mae_5_mean"]) else np.nan,
                "high_quality_flag_short_mean": float(r["high_quality_flag_short_mean"]) if pd.notna(r["high_quality_flag_short_mean"]) else np.nan,
            })
    return pd.DataFrame(rows)


def compress_detail_sheet(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return df
    if kind == "structure":
        keep = [
            "symbol", "structure_id", "L1_dt", "H1_dt", "L2_dt", "L1_px", "H1_px", "L2_px",
            "up_return", "up_bars", "up_efficiency", "up_trend_pos_ratio",
            "up_vol_z_mean_5", "up_vol_z_mean_20", "up_adx_mean", "up_adx_above_20_ratio", "up_di_spread_mean",
            "pb_return", "pb_bars", "pb_depth_vs_up", "pb_bars_vs_up", "pb_efficiency",
            "pb_trend_pos_ratio", "pb_trend_neg_ratio", "pb_vol_z_mean_5", "pb_vol_z_mean_20",
            "pb_adx_mean", "pb_adx_above_20_ratio", "pb_di_spread_mean",
        ]
    elif kind == "event":
        keep = [
            "symbol", "structure_id", "event_id", "event_idx", "trigger_dt", "trigger_px",
            "divergence_id", "divergence", "bias", "supertrend_dir",
            "trigger_ret_1d", "trigger_body_pct", "trigger_upper_shadow_pct", "trigger_lower_shadow_pct", "trigger_close_pos",
            "trigger_macd", "trigger_signal", "trigger_hist",
            "trigger_adx", "trigger_di_plus", "trigger_di_minus", "trigger_dx", "trigger_di_spread",
            "trigger_adx_above_20_flag", "trigger_di_plus_gt_di_minus_flag",
            "trigger_vol_z_5", "trigger_vol_z_20",
            "bars_from_H1_to_trigger", "bars_from_L2_to_trigger", "trigger_position_in_pb",
            "pb_hist_neg_area_vs_up_hist_pos_area", "pb_vol_z_mean_20_vs_up",
            "trend_preserved_during_pullback_flag", "trigger_trend_same_as_up_flag",
            # 事件上下文字段
            "events_in_structure", "event_rank_in_structure",
            "is_first_event_in_structure", "is_last_event_in_structure",
            "bars_since_prev_event_in_structure", "event_count_same_day",
            "symbol_event_rank_same_day", "trigger_year", "trigger_month", "trigger_ym", "trigger_weekday",
        ]
    else:
        keep = [
            "symbol", "event_id", "structure_id", "entry_dt", "entry_price",
            "ret_3", "ret_5", "mae_3", "mae_5", "rr_3", "rr_5",
            "quality_raw_short", "quality_score_short", "quality_bucket_short", "high_quality_flag_short",
            "ret_10", "ret_20", "mfe_10", "mae_10", "rr_10", "mfe_20", "mae_20", "rr_20",
        ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(6)
    return out


def export_full_detail_parquets(base_name: str, out_dir: str, structure_table: pd.DataFrame, event_table: pd.DataFrame, label_table: pd.DataFrame) -> List[str]:
    stem = os.path.splitext(os.path.basename(base_name))[0]
    paths: List[str] = []
    for name, df in [("structure_table_full", structure_table), ("event_table_full", event_table), ("label_table_full", label_table)]:
        if df is None or df.empty:
            continue
        path = os.path.join(out_dir, f"{stem}_{name}.parquet")
        df.to_parquet(path, index=False, compression="zstd")
        paths.append(path)
    return paths


def enrich_event_context(event_table: pd.DataFrame) -> pd.DataFrame:
    if event_table.empty:
        return event_table.copy()
    out = event_table.copy()
    if "trigger_dt" in out.columns:
        trig = pd.to_datetime(out["trigger_dt"])
        out["trigger_year"] = trig.dt.year
        out["trigger_month"] = trig.dt.month
        out["trigger_ym"] = trig.dt.strftime("%Y-%m")
        out["trigger_weekday"] = trig.dt.weekday
    group = out.groupby("structure_id", dropna=False)
    out["events_in_structure"] = group["event_id"].transform("count")
    out = out.sort_values([c for c in ["symbol", "structure_id", "event_idx", "event_id"] if c in out.columns]).reset_index(drop=True)
    out["event_rank_in_structure"] = out.groupby("structure_id", dropna=False).cumcount() + 1
    out["is_first_event_in_structure"] = (out["event_rank_in_structure"] == 1).astype(float)
    out["is_last_event_in_structure"] = (out["event_rank_in_structure"] == out["events_in_structure"]).astype(float)
    if "event_idx" in out.columns:
        out["bars_since_prev_event_in_structure"] = out.groupby("structure_id", dropna=False)["event_idx"].diff()
    else:
        out["bars_since_prev_event_in_structure"] = np.nan
    if "trigger_dt" in out.columns:
        out["event_count_same_day"] = out.groupby(pd.to_datetime(out["trigger_dt"]), dropna=False)["event_id"].transform("count")
        out["symbol_event_rank_same_day"] = out.groupby(["symbol", pd.to_datetime(out["trigger_dt"])], dropna=False).cumcount() + 1
    else:
        out["event_count_same_day"] = np.nan
        out["symbol_event_rank_same_day"] = np.nan
    return out


def build_model_table(structure_table: pd.DataFrame, event_table: pd.DataFrame, label_table: pd.DataFrame) -> pd.DataFrame:
    if event_table.empty:
        return pd.DataFrame()
    event_ctx = enrich_event_context(event_table)
    merged = event_ctx.copy()
    if not structure_table.empty:
        structure_cols = [c for c in structure_table.columns if c not in merged.columns or c == "structure_id"]
        merged = merged.merge(structure_table[structure_cols], on="structure_id", how="left")
    if not label_table.empty:
        label_cols = [c for c in label_table.columns if c not in merged.columns or c == "event_id"]
        merged = merged.merge(label_table[label_cols], on="event_id", how="left")
    key_feature_candidates = [
        "pb_depth_vs_up", "pb_bars_vs_up", "trigger_vol_z_5", "trigger_vol_z_20",
        "trigger_adx", "trigger_di_spread", "up_adx_mean", "pb_adx_mean",
        "quality_score_short",
    ]
    key_features = [c for c in key_feature_candidates if c in merged.columns]
    if key_features:
        merged["feature_missing_count"] = merged[key_features].isna().sum(axis=1)
        merged["has_missing_key_feature_flag"] = (merged["feature_missing_count"] > 0).astype(float)
    else:
        merged["feature_missing_count"] = 0.0
        merged["has_missing_key_feature_flag"] = 0.0
    return merged


def export_model_table(base_name: str, out_dir: str, model_table: pd.DataFrame) -> Optional[str]:
    if model_table is None or model_table.empty:
        return None
    stem = os.path.splitext(os.path.basename(base_name))[0]
    path = os.path.join(out_dir, f"{stem}_model_table_short_quality.parquet")
    model_table.to_parquet(path, index=False, compression="zstd")
    return path


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
    parser.add_argument("--excel-detail", type=str, default="compact", choices=["compact", "full"], help="Excel明细输出级别")
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
        all_dfs = fetch_data_from_db(args.n_stocks, args.bars, args.freq, args.seed, zigzag_cfg, vol_z_windows, max_workers=max(1, int(args.max_workers)))
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
        n_workers=args.prepare_workers
    )

    event_table = enrich_event_context(event_table)
    model_table = build_model_table(structure_table, event_table, label_table)

    # Merge labels back to events for analysis sheets
    merged_event_label = event_table.merge(
        label_table[[c for c in label_table.columns if c not in event_table.columns or c == "event_id"]],
        on="event_id",
        how="left",
        suffixes=("", "_label"),
    ) if not event_table.empty and not label_table.empty else event_table.copy()

    dataset_summary = summarize_dataset(structure_table, event_table, label_table)
    event_quality_summary = build_event_quality_summary(merged_event_label)
    bucket_features = [
        "pb_depth_vs_up", "pb_bars_vs_up", "pb_hist_neg_area_vs_up_hist_pos_area",
        "pb_trend_pos_ratio", "trigger_vol_z_5", "trigger_vol_z_20", "up_vol_z_mean_20", "pb_vol_z_mean_20",
        "trigger_adx", "trigger_di_spread", "trigger_adx_above_20_flag",
        "up_adx_mean", "up_adx_above_20_ratio", "pb_adx_mean", "pb_adx_above_20_ratio",
        "up_di_spread_mean", "pb_di_spread_mean",
    ]
    bucket_profiles = build_bucket_profiles(merged_event_label, bucket_features, target="quality_score_short", q=5)
    feature_inventory = build_feature_inventory(structure_table, event_table, label_table)
    model_feature_manifest = build_model_feature_manifest(structure_table, event_table, label_table)

    structure_for_excel = structure_table if args.excel_detail == "full" else compress_detail_sheet(structure_table, "structure")
    event_for_excel = event_table if args.excel_detail == "full" else compress_detail_sheet(event_table, "event")
    label_for_excel = label_table if args.excel_detail == "full" else compress_detail_sheet(label_table, "label")

    # 准备model_table预览（前1000行）
    model_table_preview = model_table.head(1000).copy() if model_table is not None and not model_table.empty else pd.DataFrame()
    
    result_tables = {
        "dataset_summary": dataset_summary,
        "structure_table": structure_for_excel,
        "event_table": event_for_excel,
        "label_table": label_for_excel,
        "event_quality_summary": event_quality_summary,
        "bucket_profiles": bucket_profiles,
        "feature_inventory": feature_inventory,
        "model_feature_manifest": model_feature_manifest,
        "model_table_preview": model_table_preview,
    }
    workbook_path = export_results_workbook(result_tables, OUT_DIR, args.excel_name)
    parquet_paths = export_full_detail_parquets(args.excel_name, OUT_DIR, structure_table, event_table, label_table)
    
    # 导出model_table用于GBDT训练
    model_table_path = export_model_table(args.excel_name, OUT_DIR, model_table)
    
    print(f"\n完成，Excel 已导出: {workbook_path}")
    if parquet_paths:
        print("完整明细 Parquet:")
        for p in parquet_paths:
            print(f"  - {p}")
    if model_table_path:
        print(f"Model table 已导出: {model_table_path}")


if __name__ == "__main__":
    main()
