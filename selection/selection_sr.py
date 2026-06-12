#!/usr/bin/env python3
"""
SR 场景化选股脚本（基于周线 SR 事件因子）

Purpose:
    基于 sr_event_factor_lab 的 SR 事件因子，识别两类可交易场景：
    1) R2S刺破收回+缩量确认（R2S_PIERCE_RECLAIM）—— 周线最优: +缩量+老R2S → ret_20=6.91%
    2) 动量回踩+不破active_support（MOMENTUM_PULLBACK）—— 周线最优: 等回踩active_support → ret=8.70%
    另有破位剔除（BREAKDOWN_EXCLUDE）。
    低吸场景(LOW_BUY_WATCH)因实验证实收益为负，已移除。

Inputs:
    stock_k_data (周线K线 w + 日线K线 d)

Outputs:
    sr_selection (选股结果表)

How to Run:
    python selection/selection_sr.py                          # 当天选股
    python selection/selection_sr.py 2025-05-16               # 指定日期
    python selection/selection_sr.py --test 300133             # 测试单只
    python selection/selection_sr.py --no-save                 # 不写入数据库
    python selection/selection_sr.py --backfill 2025-01-01 2026-04-30  # 回补历史

Side Effects:
    写入 sr_selection 表（幂等：同一日期+同一freq先删后插）

================================================================================
【选股逻辑——基于实验结论】

场景一：R2S_PIERCE_RECLAIM（R2S刺破收回+缩量确认）
  实验结论：周线R2S刺破收回+缩量 mean_ret_20=4.43%，+老R2S(>20bar)=6.91%
  触发条件（全部满足）：
    1) evt_pierce_flipped_support_reclaim == True（low跌破flipped_support但close站回）
    2) is_volume_shrink == True（缩量收回，放量收回mean_ret仅0.30%）
    3) is_support_flipped == True（确认R2S已发生）
    4) daily_broken_weekly_low == False（日线未跌破周线low）
  质量加分（记录在reason_tags，影响优先级）：
    - flipped_support_age_bars > 20 → 老R2S(实验6.91% vs 新R2S 2.06%)
    - is_shallow_support_pierce → 浅刺破（空头试探力度弱）
    - flipped_support_reclaim_strength_atr > 0.5 → 强收回
    - close_pos_in_bar >= 0.6 → 收盘上半部
    - support_cluster_is_strong → 强支撑簇
  风险标记（记录在risk_tags）：
    - risk_score > 0.8351 → 高风险(GBDT模型)
  失效条件：
    - 收盘跌破 flipped_support_ref
    - 放量跌破（is_volume_expansion + 跌破flipped_support）

场景二：MOMENTUM_PULLBACK（动量回踩+不破active_support）
  实验结论：等回踩active_support不破 → mean_ret=8.70%, win_rate=66.5%, plr=3.04
  触发条件（全部满足）：
    1) evt_pierce_support_cluster_reclaim_low_volume == True（强簇缩量收回）
    2) ret_since_signal > 3%（已反弹，确认动量建立）
    3) daily_broken_weekly_low == False（日线未跌破周线low）
    4) close > active_support_ref（当前价在active_support之上）
  质量加分：
    - support_cluster_is_strong → 强支撑簇
    - manual_rr >= 1.5 → 盈亏比合理
    - close_pos_in_bar >= 0.5 → 收盘位置偏上
  风险标记：
    - risk_score > 0.8351 → 高风险
  失效条件：
    - 放量跌破 active_support_ref
    - 跌破周线low

场景三：BREAKDOWN_EXCLUDE（剔除）
  触发条件（满足任一）：
    1) daily_broken_weekly_low == True（日线跌破周线low）
    2) evt_breakdown_flipped_support == True（R2S跌破）
    3) evt_pierce_support_cluster_reclaim_high_volume == True（放量刺破）

【核心计算】
  - 全部引用 features.sr_event_factor_lab.compute_sr_factor_lab（SSOT）
  - 数据加载引用 sr_experiment.db_adapter
  - 本脚本只做数据准备、场景分类、结果保存，不重复计算逻辑

【选股日期】
  - 选股日期只是标记，实际数据到"选股日期当天或之前最后一个交易日"
  - 周线数据通过 python app/build_dataset.py --update --period w 每周更新
  - 当前仅支持周线选股（freq='w'），日线选股预留扩展

【保存逻辑】
  - 按选股日期+freq统一保存，先删旧数据再插新数据（幂等性）
================================================================================
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Optional, List
from tqdm import tqdm

from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig
from sr_experiment.db_adapter import get_stock_pool, load_kline
from sr_experiment.gbdt_feature_columns import ALL_FEATURE_COLS

try:
    import lightgbm as lgb
    from sr_experiment.gbdt_config import MODELS_DIR
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "sr_selection"

WEEKLY_BARS = 300
DAILY_BARS = 800

SCENARIO_TYPES = ["R2S_PIERCE_RECLAIM", "MOMENTUM_PULLBACK", "BREAKDOWN_EXCLUDE"]

RISK_THRESHOLD = 0.8351


def _predict_risk_score(factor_row: pd.Series) -> Optional[float]:
    if not _LGB_AVAILABLE:
        return None
    model_path = os.path.join(MODELS_DIR, "B_cluster_risk.txt")
    if not os.path.exists(model_path):
        return None
    try:
        model = lgb.Booster(model_file=model_path)
        feature_cols = [c for c in ALL_FEATURE_COLS if c in factor_row.index]
        if len(feature_cols) < 10:
            return None
        row_values = []
        for c in feature_cols:
            v = factor_row[c]
            row_values.append(0.0 if pd.isna(v) else float(v))
        row_df = pd.DataFrame([row_values], columns=feature_cols)
        return float(model.predict(row_df, predict_disable_shape_check=True)[0])
    except Exception:
        return None


def _get_daily_status(weekly_row: pd.Series, daily_df: pd.DataFrame) -> dict:
    w_low = weekly_row.get("low", np.nan)
    w_close = weekly_row.get("close", np.nan)
    w_time = weekly_row.name if isinstance(weekly_row.name, pd.Timestamp) else weekly_row.get("bar_time", pd.NaT)

    if daily_df.empty or pd.isna(w_time):
        return {
            "daily_broken_weekly_low": False,
            "distance_to_weekly_low_pct": np.nan,
            "ret_since_signal": np.nan,
            "current_to_resistance_pct": np.nan,
            "latest_daily_close": np.nan,
        }

    daily_after = daily_df[daily_df.index > w_time]
    if daily_after.empty:
        latest = daily_df.iloc[-1]
    else:
        latest = daily_after.iloc[-1]

    latest_close = latest.get("close", np.nan)

    broken = False
    if not daily_after.empty:
        for _, d_row in daily_after.iterrows():
            d_low = d_row.get("low", np.nan)
            if pd.notna(w_low) and pd.notna(d_low) and d_low < w_low:
                broken = True
                break

    dist_pct = (latest_close - w_low) / w_low * 100 if pd.notna(latest_close) and pd.notna(w_low) and w_low > 0 else np.nan
    ret_since = (latest_close - w_close) / w_close * 100 if pd.notna(latest_close) and pd.notna(w_close) and w_close > 0 else np.nan

    resistance = weekly_row.get("resistance_ref", np.nan)
    if pd.isna(resistance):
        resistance = weekly_row.get("resistance_active", np.nan)
    dist_resistance = (resistance - latest_close) / latest_close * 100 if pd.notna(resistance) and pd.notna(latest_close) and latest_close > 0 else np.nan

    return {
        "daily_broken_weekly_low": broken,
        "distance_to_weekly_low_pct": dist_pct,
        "ret_since_signal": ret_since,
        "current_to_resistance_pct": dist_resistance,
        "latest_daily_close": latest_close,
    }


def _compute_manual_rr(row: pd.Series, daily_status: dict) -> Optional[float]:
    close = daily_status.get("latest_daily_close", np.nan)
    if pd.isna(close) or close <= 0:
        return None
    resistance = row.get("resistance_ref", np.nan)
    if pd.isna(resistance):
        resistance = row.get("resistance_active", np.nan)
    if pd.isna(resistance) or resistance <= 0:
        return None
    w_low = row.get("low", np.nan)
    if pd.isna(w_low) or w_low <= 0:
        return None
    upside = resistance / close - 1
    downside = close / w_low - 1
    if downside <= 0:
        return None
    return upside / downside


def _classify_scenario(row: pd.Series, daily_status: dict, risk_score: Optional[float] = None) -> dict:
    broken = daily_status.get("daily_broken_weekly_low", False)
    ret_since = daily_status.get("ret_since_signal", np.nan)
    latest_close = daily_status.get("latest_daily_close", np.nan)

    positive_tags = []
    negative_tags = []

    if bool(row.get("is_support_flipped", False)):
        positive_tags.append("R2S已验证")
    if bool(row.get("support_cluster_is_strong", False)):
        positive_tags.append("强支撑簇")
    if bool(row.get("is_volume_shrink", False)):
        positive_tags.append("缩量")
    if not broken:
        positive_tags.append("未破周线low")
    if pd.notna(row.get("close_pos_in_bar")) and row["close_pos_in_bar"] >= 0.6:
        positive_tags.append("收盘上半部")

    if broken:
        negative_tags.append("日线跌破周线low")
    if bool(row.get("evt_pierce_support_cluster_reclaim_high_volume", False)):
        negative_tags.append("放量刺破")
    if bool(row.get("evt_breakdown_flipped_support", False)):
        negative_tags.append("R2S跌破")
    if pd.notna(ret_since) and ret_since > 10:
        negative_tags.append(f"涨幅过大({ret_since:.1f}%)")
    if risk_score is not None and risk_score > RISK_THRESHOLD:
        negative_tags.append(f"高风险({risk_score:.3f})")

    if broken or bool(row.get("evt_breakdown_flipped_support", False)) or bool(row.get("evt_pierce_support_cluster_reclaim_high_volume", False)):
        scenario_type = "BREAKDOWN_EXCLUDE"
        action = "已破位/放量刺破，剔除"
        invalid = "已失效"
    elif (bool(row.get("evt_pierce_flipped_support_reclaim", False))
          and bool(row.get("is_volume_shrink", False))
          and bool(row.get("is_support_flipped", False))
          and not broken):
        scenario_type = "R2S_PIERCE_RECLAIM"
        age = row.get("flipped_support_age_bars", np.nan)
        age_str = f"({int(age)}bar)" if pd.notna(age) else ""
        flipped_ref = row.get("flipped_support_ref", np.nan)
        stop_str = f"{flipped_ref:.2f}" if pd.notna(flipped_ref) else "flipped_support"
        action = f"R2S刺破后缩量收回{age_str}，空头试探失败；止损={stop_str}"
        invalid = f"收盘跌破 {stop_str} 或放量跌破"
        if pd.notna(age) and age > 20:
            positive_tags.append("老R2S(>20bar)")
        if bool(row.get("is_shallow_support_pierce", False)):
            positive_tags.append("浅刺破")
        reclaim_str = row.get("flipped_support_reclaim_strength_atr", np.nan)
        if pd.notna(reclaim_str) and reclaim_str > 0.5:
            positive_tags.append(f"强收回({reclaim_str:.2f}ATR)")
    elif (bool(row.get("evt_pierce_support_cluster_reclaim_low_volume", False))
          and pd.notna(ret_since) and ret_since > 3
          and not broken
          and pd.notna(latest_close) and pd.notna(row.get("active_support_ref"))
          and latest_close > row["active_support_ref"]):
        scenario_type = "MOMENTUM_PULLBACK"
        active_sup = row.get("active_support_ref", np.nan)
        stop_str = f"{active_sup:.2f}" if pd.notna(active_sup) else "active_support"
        action = f"强簇缩量收回已反弹{ret_since:.1f}%，等回踩{stop_str}不破再入"
        invalid = f"放量跌破 {stop_str} 或跌破周线low"
        manual_rr = _compute_manual_rr(row, daily_status)
        if pd.notna(manual_rr) and manual_rr >= 1.5:
            positive_tags.append(f"盈亏比{manual_rr:.1f}")
        if pd.notna(row.get("close_pos_in_bar")) and row["close_pos_in_bar"] >= 0.5:
            positive_tags.append("收盘偏上")
    else:
        return {
            "scenario_type": None,
            "action_suggestion": "不满足任何场景",
            "invalid_condition": "",
            "reason_tags": "; ".join(positive_tags),
            "risk_tags": "; ".join(negative_tags),
        }

    return {
        "scenario_type": scenario_type,
        "action_suggestion": action,
        "invalid_condition": invalid,
        "reason_tags": "; ".join(positive_tags),
        "risk_tags": "; ".join(negative_tags),
    }


def process_stock(ts_code: str, selection_date: date, freq: str = "w", pivot_len: int = 10) -> Optional[Dict]:
    weekly_df = load_kline(ts_code, freq, end_date=selection_date)
    if weekly_df is None or weekly_df.empty or len(weekly_df) < 30:
        return None

    cfg = LabConfig(pivot_len=pivot_len)
    try:
        result_df = compute_sr_factor_lab(weekly_df, cfg)
    except Exception:
        return None

    if result_df.empty:
        return None

    last = result_df.iloc[-1]

    has_event = False
    for evt in ["evt_pierce_flipped_support_reclaim", "evt_retest_flipped_support",
                "evt_pierce_support_cluster_reclaim_low_volume",
                "evt_breakdown_flipped_support", "evt_pierce_support_cluster_reclaim_high_volume"]:
        if bool(last.get(evt, False)):
            has_event = True
            break

    if not has_event:
        return None

    daily_df = load_kline(ts_code, "d", end_date=selection_date)
    daily_status = _get_daily_status(last, daily_df if daily_df is not None else pd.DataFrame())

    risk_score = _predict_risk_score(last)
    manual_rr = _compute_manual_rr(last, daily_status)

    classification = _classify_scenario(last, daily_status, risk_score=risk_score)
    if classification["scenario_type"] is None:
        return None

    signal_date = last.name if isinstance(last.name, pd.Timestamp) else last.get("bar_time", None)

    return {
        "ts_code": ts_code,
        "signal_date": signal_date,
        "freq": freq,
        "scenario_type": classification["scenario_type"],
        "action_suggestion": classification["action_suggestion"],
        "invalid_condition": classification["invalid_condition"],
        "reason_tags": classification["reason_tags"],
        "risk_tags": classification["risk_tags"],
        "is_support_flipped": bool(last.get("is_support_flipped", False)),
        "flipped_support_ref": float(last["flipped_support_ref"]) if pd.notna(last.get("flipped_support_ref")) else None,
        "flipped_support_age_bars": int(last["flipped_support_age_bars"]) if pd.notna(last.get("flipped_support_age_bars")) else None,
        "evt_pierce_flipped_support_reclaim": bool(last.get("evt_pierce_flipped_support_reclaim", False)),
        "flipped_support_reclaim_strength_atr": float(last["flipped_support_reclaim_strength_atr"]) if pd.notna(last.get("flipped_support_reclaim_strength_atr")) else None,
        "active_support_ref": float(last["active_support_ref"]) if pd.notna(last.get("active_support_ref")) else None,
        "resistance_ref": float(last["resistance_ref"]) if pd.notna(last.get("resistance_ref")) else None,
        "support_cluster_is_strong": bool(last.get("support_cluster_is_strong", False)),
        "support_cluster_score": float(last["support_cluster_score"]) if pd.notna(last.get("support_cluster_score")) else None,
        "support_confluence_score": float(last["support_confluence_score"]) if pd.notna(last.get("support_confluence_score")) else None,
        "is_volume_shrink": bool(last.get("is_volume_shrink", False)),
        "volume_z_20": float(last["volume_z_20"]) if pd.notna(last.get("volume_z_20")) else None,
        "close_pos_in_bar": float(last["close_pos_in_bar"]) if pd.notna(last.get("close_pos_in_bar")) else None,
        "support_reclaim_strength_atr": float(last["support_reclaim_strength_atr"]) if pd.notna(last.get("support_reclaim_strength_atr")) else None,
        "risk_score": risk_score,
        "manual_rr": manual_rr,
        "distance_to_weekly_low_pct": daily_status["distance_to_weekly_low_pct"],
        "current_to_resistance_pct": daily_status["current_to_resistance_pct"],
        "ret_since_signal": daily_status["ret_since_signal"],
        "daily_broken_weekly_low": daily_status["daily_broken_weekly_low"],
        "weekly_close": float(last["close"]) if pd.notna(last.get("close")) else None,
        "weekly_low": float(last["low"]) if pd.notna(last.get("low")) else None,
    }


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    if not ts_codes:
        return {}
    try:
        pool_df = get_stock_pool()
        name_map = dict(zip(pool_df["ts_code"], pool_df["name"]))
        result = {}
        for code in ts_codes:
            result[code] = name_map.get(code, name_map.get(code.replace(".SZ", "").replace(".SH", ""), ""))
        return result
    except Exception:
        placeholders = ", ".join([f"'{c}'" for c in ts_codes])
        sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
        with engine.connect() as conn:
            result = conn.execute(sql)
            return {row[0]: row[1] for row in result}


def ensure_table_exists():
    create_sql = """
    CREATE TABLE IF NOT EXISTS sr_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date TIMESTAMP,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),
        freq VARCHAR(5) DEFAULT 'w',
        scenario_type VARCHAR(30) NOT NULL,
        action_suggestion TEXT,
        invalid_condition TEXT,
        reason_tags TEXT,
        risk_tags TEXT,
        is_support_flipped BOOLEAN DEFAULT FALSE,
        flipped_support_ref FLOAT,
        flipped_support_age_bars INT,
        evt_pierce_flipped_support_reclaim BOOLEAN DEFAULT FALSE,
        flipped_support_reclaim_strength_atr FLOAT,
        active_support_ref FLOAT,
        resistance_ref FLOAT,
        support_cluster_is_strong BOOLEAN DEFAULT FALSE,
        support_cluster_score FLOAT,
        support_confluence_score FLOAT,
        is_volume_shrink BOOLEAN DEFAULT FALSE,
        volume_z_20 FLOAT,
        close_pos_in_bar FLOAT,
        support_reclaim_strength_atr FLOAT,
        risk_score FLOAT,
        manual_rr FLOAT,
        distance_to_weekly_low_pct FLOAT,
        current_to_resistance_pct FLOAT,
        ret_since_signal FLOAT,
        daily_broken_weekly_low BOOLEAN DEFAULT FALSE,
        weekly_close FLOAT,
        weekly_low FLOAT,
        change_pct FLOAT,
        batch_no INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code, freq)
    );
    CREATE INDEX IF NOT EXISTS idx_sr_selection_date ON sr_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_sr_ts_code ON sr_selection(ts_code);
    CREATE INDEX IF NOT EXISTS idx_sr_scenario_type ON sr_selection(scenario_type);
    CREATE INDEX IF NOT EXISTS idx_sr_freq ON sr_selection(freq);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()


def save_to_database(df: pd.DataFrame, selection_date: date, freq: str = "w") -> int:
    if df.empty:
        print("数据为空，跳过数据库保存")
        return 0

    ensure_table_exists()

    with engine.connect() as conn:
        delete_sql = text(f"DELETE FROM {SELECTION_TABLE} WHERE selection_date = :selection_date AND freq = :freq")
        result = conn.execute(delete_sql, {"selection_date": selection_date, "freq": freq})
        conn.commit()
        if result.rowcount > 0:
            print(f"  清除旧数据: {result.rowcount} 条")

    records = []
    for _, row in df.iterrows():
        record = {
            "selection_date": selection_date,
            "signal_date": row.get("signal_date"),
            "ts_code": row["ts_code"],
            "stock_name": row.get("stock_name", ""),
            "freq": freq,
            "scenario_type": row["scenario_type"],
            "action_suggestion": row.get("action_suggestion", ""),
            "invalid_condition": row.get("invalid_condition", ""),
            "reason_tags": row.get("reason_tags", ""),
            "risk_tags": row.get("risk_tags", ""),
            "is_support_flipped": bool(row.get("is_support_flipped", False)),
            "flipped_support_ref": float(row["flipped_support_ref"]) if pd.notna(row.get("flipped_support_ref")) else None,
            "flipped_support_age_bars": int(row["flipped_support_age_bars"]) if pd.notna(row.get("flipped_support_age_bars")) else None,
            "evt_pierce_flipped_support_reclaim": bool(row.get("evt_pierce_flipped_support_reclaim", False)),
            "flipped_support_reclaim_strength_atr": float(row["flipped_support_reclaim_strength_atr"]) if pd.notna(row.get("flipped_support_reclaim_strength_atr")) else None,
            "active_support_ref": float(row["active_support_ref"]) if pd.notna(row.get("active_support_ref")) else None,
            "resistance_ref": float(row["resistance_ref"]) if pd.notna(row.get("resistance_ref")) else None,
            "support_cluster_is_strong": bool(row.get("support_cluster_is_strong", False)),
            "support_cluster_score": float(row["support_cluster_score"]) if pd.notna(row.get("support_cluster_score")) else None,
            "support_confluence_score": float(row["support_confluence_score"]) if pd.notna(row.get("support_confluence_score")) else None,
            "is_volume_shrink": bool(row.get("is_volume_shrink", False)),
            "volume_z_20": float(row["volume_z_20"]) if pd.notna(row.get("volume_z_20")) else None,
            "close_pos_in_bar": float(row["close_pos_in_bar"]) if pd.notna(row.get("close_pos_in_bar")) else None,
            "support_reclaim_strength_atr": float(row["support_reclaim_strength_atr"]) if pd.notna(row.get("support_reclaim_strength_atr")) else None,
            "risk_score": float(row["risk_score"]) if pd.notna(row.get("risk_score")) else None,
            "manual_rr": float(row["manual_rr"]) if pd.notna(row.get("manual_rr")) else None,
            "distance_to_weekly_low_pct": float(row["distance_to_weekly_low_pct"]) if pd.notna(row.get("distance_to_weekly_low_pct")) else None,
            "current_to_resistance_pct": float(row["current_to_resistance_pct"]) if pd.notna(row.get("current_to_resistance_pct")) else None,
            "ret_since_signal": float(row["ret_since_signal"]) if pd.notna(row.get("ret_since_signal")) else None,
            "daily_broken_weekly_low": bool(row.get("daily_broken_weekly_low", False)),
            "weekly_close": float(row["weekly_close"]) if pd.notna(row.get("weekly_close")) else None,
            "weekly_low": float(row["weekly_low"]) if pd.notna(row.get("weekly_low")) else None,
            "change_pct": float(row["change_pct"]) if pd.notna(row.get("change_pct")) else None,
            "batch_no": int(row["batch_no"]) if pd.notna(row.get("batch_no")) else None,
        }
        records.append(record)

    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists="append", index=False)
        print(f"  保存新数据: {len(records)} 条")
        return len(records)

    return 0


def select_sr_stocks(selection_date: Optional[date] = None, save_to_db: bool = True,
                     freq: str = "w", pivot_len: int = 10) -> pd.DataFrame:
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("SR 场景化选股：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  周期: {freq}")
    print(f"  pivot_len: {pivot_len}")
    print(f"  场景: R2S_PIERCE_RECLAIM / MOMENTUM_PULLBACK / BREAKDOWN_EXCLUDE")
    print("=" * 80)

    with engine.connect() as conn:
        print("\n查询所有股票...")
        sql = text("""
            SELECT DISTINCT ts_code
            FROM stock_k_data
            WHERE freq = :freq AND DATE(bar_time) <= :end_date
        """)
        stock_list = pd.read_sql(sql, conn, params={
            "freq": freq,
            "end_date": selection_date.strftime("%Y-%m-%d"),
        })
        print(f"  找到 {len(stock_list)} 只股票")

    if len(stock_list) == 0:
        print("未找到符合条件的股票")
        return pd.DataFrame()

    print(f"\n开始 SR 场景化筛选...")

    filtered_results = []
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="SR选股", unit="只"):
        ts_code = row["ts_code"]
        result = process_stock(ts_code, selection_date, freq=freq, pivot_len=pivot_len)
        if result:
            filtered_results.append(result)

    result_df = pd.DataFrame(filtered_results)

    if not result_df.empty:
        stock_names = batch_get_stock_names(result_df["ts_code"].tolist())
        result_df["stock_name"] = result_df["ts_code"].map(stock_names)
        result_df["batch_no"] = (result_df.index // 10) + 1

    print("\n" + "=" * 80)
    print("选股结果汇总：")
    print("=" * 80)
    print(f"SR筛选后: {len(result_df)} 只")

    if not result_df.empty:
        print(f"\n场景统计：")
        for st in SCENARIO_TYPES:
            count = (result_df["scenario_type"] == st).sum()
            print(f"  {st}: {count} 只")

        print(f"\n前20名股票：")
        display_cols = ["ts_code", "stock_name", "scenario_type", "action_suggestion", "reason_tags"]
        print_cols = [c for c in display_cols if c in result_df.columns]
        print(result_df[print_cols].head(20).to_string(index=False))

    if save_to_db:
        print("\n" + "-" * 80)
        print("保存到数据库...")
        saved_count = save_to_database(result_df, selection_date, freq=freq)
        print("-" * 80)

    return result_df


def backfill_stock_events(ts_code: str, start_date: date, end_date: date,
                          freq: str = "w", pivot_len: int = 10) -> List[Dict]:
    weekly_df = load_kline(ts_code, freq, end_date=end_date)
    if weekly_df is None or weekly_df.empty or len(weekly_df) < 30:
        return []

    cfg = LabConfig(pivot_len=pivot_len)
    try:
        result_df = compute_sr_factor_lab(weekly_df, cfg)
    except Exception:
        return []

    if result_df.empty:
        return []

    mask = (result_df.index >= pd.Timestamp(start_date)) & (result_df.index <= pd.Timestamp(end_date))
    trade_bars = result_df.loc[mask]
    if trade_bars.empty:
        return []

    daily_df = load_kline(ts_code, "d", end_date=end_date)
    daily_df_empty = daily_df is None or daily_df.empty

    results = []
    for bar_time, row in trade_bars.iterrows():
        has_event = False
        for evt in ["evt_pierce_flipped_support_reclaim", "evt_retest_flipped_support",
                    "evt_pierce_support_cluster_reclaim_low_volume",
                    "evt_breakdown_flipped_support", "evt_pierce_support_cluster_reclaim_high_volume"]:
            if bool(row.get(evt, False)):
                has_event = True
                break

        if not has_event:
            continue

        daily_status = _get_daily_status(row, daily_df if not daily_df_empty else pd.DataFrame())
        risk_score = _predict_risk_score(row)
        manual_rr = _compute_manual_rr(row, daily_status)
        classification = _classify_scenario(row, daily_status, risk_score=risk_score)
        if classification["scenario_type"] is None:
            continue

        results.append({
            "ts_code": ts_code,
            "signal_date": bar_time,
            "freq": freq,
            "scenario_type": classification["scenario_type"],
            "action_suggestion": classification["action_suggestion"],
            "invalid_condition": classification["invalid_condition"],
            "reason_tags": classification["reason_tags"],
            "risk_tags": classification["risk_tags"],
            "is_support_flipped": bool(row.get("is_support_flipped", False)),
            "flipped_support_ref": float(row["flipped_support_ref"]) if pd.notna(row.get("flipped_support_ref")) else None,
            "flipped_support_age_bars": int(row["flipped_support_age_bars"]) if pd.notna(row.get("flipped_support_age_bars")) else None,
            "evt_pierce_flipped_support_reclaim": bool(row.get("evt_pierce_flipped_support_reclaim", False)),
            "flipped_support_reclaim_strength_atr": float(row["flipped_support_reclaim_strength_atr"]) if pd.notna(row.get("flipped_support_reclaim_strength_atr")) else None,
            "active_support_ref": float(row["active_support_ref"]) if pd.notna(row.get("active_support_ref")) else None,
            "resistance_ref": float(row["resistance_ref"]) if pd.notna(row.get("resistance_ref")) else None,
            "support_cluster_is_strong": bool(row.get("support_cluster_is_strong", False)),
            "support_cluster_score": float(row["support_cluster_score"]) if pd.notna(row.get("support_cluster_score")) else None,
            "support_confluence_score": float(row["support_confluence_score"]) if pd.notna(row.get("support_confluence_score")) else None,
            "is_volume_shrink": bool(row.get("is_volume_shrink", False)),
            "volume_z_20": float(row["volume_z_20"]) if pd.notna(row.get("volume_z_20")) else None,
            "close_pos_in_bar": float(row["close_pos_in_bar"]) if pd.notna(row.get("close_pos_in_bar")) else None,
            "support_reclaim_strength_atr": float(row["support_reclaim_strength_atr"]) if pd.notna(row.get("support_reclaim_strength_atr")) else None,
            "risk_score": risk_score,
            "manual_rr": manual_rr,
            "distance_to_weekly_low_pct": daily_status["distance_to_weekly_low_pct"],
            "current_to_resistance_pct": daily_status["current_to_resistance_pct"],
            "ret_since_signal": daily_status["ret_since_signal"],
            "daily_broken_weekly_low": daily_status["daily_broken_weekly_low"],
            "weekly_close": float(row["close"]) if pd.notna(row.get("close")) else None,
            "weekly_low": float(row["low"]) if pd.notna(row.get("low")) else None,
        })

    return results


def _save_single_stock_records(records: List[Dict], stock_name_map: Dict[str, str], freq: str = "w"):
    if not records:
        return 0

    ensure_table_exists()

    date_groups = defaultdict(list)
    for rec in records:
        dt = rec["signal_date"]
        if isinstance(dt, pd.Timestamp):
            dt = dt.date()
        date_groups[dt].append(rec)

    total_saved = 0
    for dt, day_records in date_groups.items():
        ts_codes = [r["ts_code"] for r in day_records]
        placeholders = ", ".join([f"'{c}'" for c in ts_codes])
        with engine.connect() as conn:
            delete_sql = text(
                f"DELETE FROM {SELECTION_TABLE} "
                f"WHERE selection_date = :selection_date AND ts_code IN ({placeholders}) AND freq = :freq"
            )
            conn.execute(delete_sql, {"selection_date": dt, "freq": freq})
            conn.commit()

        insert_records = []
        for rec in day_records:
            insert_records.append({
                "selection_date": dt,
                "signal_date": rec["signal_date"],
                "ts_code": rec["ts_code"],
                "stock_name": stock_name_map.get(rec["ts_code"], ""),
                "freq": freq,
                "scenario_type": rec["scenario_type"],
                "action_suggestion": rec.get("action_suggestion", ""),
                "invalid_condition": rec.get("invalid_condition", ""),
                "reason_tags": rec.get("reason_tags", ""),
                "risk_tags": rec.get("risk_tags", ""),
                "is_support_flipped": bool(rec.get("is_support_flipped", False)),
                "flipped_support_ref": rec.get("flipped_support_ref"),
                "flipped_support_age_bars": rec.get("flipped_support_age_bars"),
                "evt_pierce_flipped_support_reclaim": bool(rec.get("evt_pierce_flipped_support_reclaim", False)),
                "flipped_support_reclaim_strength_atr": rec.get("flipped_support_reclaim_strength_atr"),
                "active_support_ref": rec.get("active_support_ref"),
                "resistance_ref": rec.get("resistance_ref"),
                "support_cluster_is_strong": bool(rec.get("support_cluster_is_strong", False)),
                "support_cluster_score": rec.get("support_cluster_score"),
                "support_confluence_score": rec.get("support_confluence_score"),
                "is_volume_shrink": bool(rec.get("is_volume_shrink", False)),
                "volume_z_20": rec.get("volume_z_20"),
                "close_pos_in_bar": rec.get("close_pos_in_bar"),
                "support_reclaim_strength_atr": rec.get("support_reclaim_strength_atr"),
                "risk_score": rec.get("risk_score"),
                "manual_rr": rec.get("manual_rr"),
                "distance_to_weekly_low_pct": rec.get("distance_to_weekly_low_pct"),
                "current_to_resistance_pct": rec.get("current_to_resistance_pct"),
                "ret_since_signal": rec.get("ret_since_signal"),
                "daily_broken_weekly_low": bool(rec.get("daily_broken_weekly_low", False)),
                "weekly_close": rec.get("weekly_close"),
                "weekly_low": rec.get("weekly_low"),
                "batch_no": None,
            })

        if insert_records:
            insert_df = pd.DataFrame(insert_records)
            insert_df.to_sql(SELECTION_TABLE, engine, if_exists="append", index=False)
            total_saved += len(insert_records)

    return total_saved


def backfill_range(start_date: date, end_date: date, stock_list: Optional[List[str]] = None,
                   freq: str = "w", pivot_len: int = 10):
    print("=" * 80)
    print("SR 场景化选股回补")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  周期: {freq}")
    print(f"  模式: 遍历个股，逐周计算事件，每只股票立即保存")
    print("=" * 80)

    if stock_list is None:
        with engine.connect() as conn:
            print("\n查询股票列表...")
            sql = text("""
                SELECT DISTINCT ts_code
                FROM stock_k_data
                WHERE freq = :freq
                AND DATE(bar_time) BETWEEN :start_date AND :end_date
            """)
            df = pd.read_sql(sql, conn, params={
                "freq": freq,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            })
            stock_list = df["ts_code"].tolist()
            print(f"  找到 {len(stock_list)} 只股票")

    if not stock_list:
        print("无股票可处理")
        return

    total_saved = 0
    total_stocks_with_events = 0

    for ts_code in tqdm(stock_list, desc="回补个股", unit="只"):
        records = backfill_stock_events(ts_code, start_date, end_date, freq=freq, pivot_len=pivot_len)
        if records:
            stock_name_map = batch_get_stock_names([ts_code])
            saved = _save_single_stock_records(records, stock_name_map, freq=freq)
            total_saved += saved
            total_stocks_with_events += 1

    print("-" * 80)
    print(f"\n回补完成")
    print(f"共处理 {len(stock_list)} 只股票，其中 {total_stocks_with_events} 只有触发事件")
    print(f"共保存 {total_saved} 条记录")
    print(f"日期范围: {start_date} ~ {end_date}")


def parse_date(date_str: str) -> date:
    for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def test_single_stock(ts_code: str, selection_date: date, freq: str = "w", pivot_len: int = 10):
    print("\n" + "=" * 80)
    print(f"测试单只股票: {ts_code}")
    print(f"选股日期: {selection_date}")
    print(f"周期: {freq}")
    print("=" * 80)

    result = process_stock(ts_code, selection_date, freq=freq, pivot_len=pivot_len)

    if result:
        print("\n选股结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("\n该股票不满足任何SR场景条件")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SR 场景化选股工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_sr.py                              # 当天选股
  python selection/selection_sr.py 2026-05-16                   # 指定日期
  python selection/selection_sr.py --test 300133                # 测试单只
  python selection/selection_sr.py --no-save                    # 不写入数据库
  python selection/selection_sr.py --backfill 2025-01-01 2026-04-30  # 回补历史
        """,
    )
    parser.add_argument("date", nargs="?", help="选股日期 (YYYY-MM-DD 或 YYYYMMDD)，默认当天")
    parser.add_argument("--test", help="测试单只股票，例如: --test 300133")
    parser.add_argument("--no-save", action="store_true", help="不保存到数据库")
    parser.add_argument("--backfill", nargs=2, metavar=("START_DATE", "END_DATE"),
                        help="回补历史事件，例如: --backfill 2025-01-01 2026-04-30")
    parser.add_argument("--freq", default="w", choices=["w", "d"], help="周期 (w=周线, d=日线预留)")
    parser.add_argument("--pivot-len", type=int, default=10, help="pivot确认长度 (默认10)")

    args = parser.parse_args()

    if args.backfill:
        start_date = parse_date(args.backfill[0])
        end_date = parse_date(args.backfill[1])
        backfill_range(start_date, end_date, freq=args.freq, pivot_len=args.pivot_len)
        sys.exit(0)

    if args.date:
        try:
            selection_date = parse_date(args.date)
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
    else:
        selection_date = date.today()

    print("\n" + "=" * 80)
    print("SR 场景化选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"周期: {args.freq}")
    print("=" * 80)

    if args.test:
        test_single_stock(args.test, selection_date, freq=args.freq, pivot_len=args.pivot_len)
        sys.exit(0)

    df = select_sr_stocks(selection_date=selection_date, save_to_db=not args.no_save,
                          freq=args.freq, pivot_len=args.pivot_len)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}' AND freq = '{args.freq}'")
    print("=" * 80)

    return df


if __name__ == "__main__":
    main()
