#!/usr/bin/env python3
"""
每日推理生成器 — 唯一负责从原始数据构造当日候选 + 持仓股 → 加载模型推理 → 生成 predictions/{T}.parquet。

用途：
    模型低频训练（按周/月），推理高频生成（每日盘后自动）。
    不依赖 candidate_with_scores.parquet（那是研究产物），不从研究表切一天。
    候选池股票（有SLC信号）与持仓股（无SLC信号）在同一 DataFrame 中统一计算4模型输出。

用法：
    # 单日推理（默认含持仓股）
    python -m stop_experiment.pipeline.07_generate_daily_predictions --date 2026-05-13
    python -m stop_experiment.pipeline.07_generate_daily_predictions --date 2026-05-13 --force

    # 单日推理（不含持仓股）
    python -m stop_experiment.pipeline.07_generate_daily_predictions --date 2026-05-13 --no-positions

    # 批量回补
    python -m stop_experiment.pipeline.07_generate_daily_predictions --start-date 2026-01-01 --end-date 2026-05-11
    python -m stop_experiment.pipeline.07_generate_daily_predictions --start-date 2026-01-01 --end-date 2026-05-11 --force

参数：
    --date           目标交易日 YYYY-MM-DD（不指定则自动取最新交易日）
    --start-date     回补起始日期 YYYY-MM-DD（与 --end-date 配合使用）
    --end-date       回补结束日期 YYYY-MM-DD（与 --start-date 配合使用）
    --force          强制覆盖已存在的预测账本
    --no-positions   禁用持仓股模式（默认启用持仓股）

流程（6步）：
    Step 1: 解析目标交易日 T
    Step 2: 从 DB 构造 obs_date=T 的单日候选样本（含持仓股合并）
    Step 3: 拼接因子库特征 + 派生特征（统一路径）
    Step 4: 加载已训练 final 模型（4个）
    Step 5: 推理 + score_stocks 打分
    Step 6: 写出 predictions/T.parquet

输入：
    - DB: stop_loss_selection (信号), stock_k_data (K线)
    - output/models_control/xxx_final.txt (4个已训练模型)
    - QMT: 持仓列表（--positions 模式）

输出：
    - output/predictions/YYYY-MM-DD.parquet
      列: ts_code, signal_id, obs_date, obs_day, pred_sell_reg, pred_sell_cls,
           pred_buy_reg, pred_buy_cls, score
    - DB: stop_loss_predictions (profile=production，持仓股 signal_id=-1)

副作用：只读 DB + parquet，写 parquet + DB（幂等）
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sqlalchemy import text

from datasource.database import get_engine
from stop_experiment.pipeline.stop_config import (
    OBS_DAYS, SELL_CLS_THRESHOLD, BUY_CLS_THRESHOLD,
    OUTPUT_DIR, MODELS_DIR, PREDICTIONS_DIR, MODEL_SPECS,
    PRODUCTION_PARAMS, CANDIDATE_OBS_DAYS,
)
from stop_experiment.pipeline.factor_columns import (
    SLC_STATIC_COLS, ALL_FEATURE_COLS, META_COLS,
)
from stop_experiment.backtest.simple_backtest import score_stocks


SIGNAL_LOOKBACK_DAYS = OBS_DAYS + 5
OBS_COLS = ["ts_code", "stock_name", "signal_id", "obs_date", "obs_day",
            "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls", "score",
            "can_buy"]


# ==================== Step 1: 日期解析 ====================

def _get_trading_days(start_date, end_date, engine):
    """从 DB stock_k_data 获取日期范围内的交易日列表。返回 pd.DatetimeIndex。"""
    sd = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    ed = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    with engine.connect() as conn:
        sql = text(
            "SELECT DISTINCT bar_time FROM stock_k_data "
            "WHERE freq = 'd' AND bar_time >= :sd AND bar_time <= :ed "
            "ORDER BY bar_time"
        )
        dates = pd.read_sql(sql, conn, params={"sd": sd, "ed": ed})["bar_time"]
    return pd.DatetimeIndex(dates.sort_values())


def _resolve_trade_date(args, engine):
    """确定目标交易日。--date 指定 → 使用；否则从 K 线取最新。"""
    if args.date:
        return pd.to_datetime(args.date)

    with engine.connect() as conn:
        sql = "SELECT MAX(bar_time) FROM stock_k_data WHERE freq = 'd'"
        result = conn.execute(text(sql)).scalar()
        if result is None:
            raise RuntimeError("无法获取最新交易日，stock_k_data 为空")
        latest = pd.to_datetime(result)
    print(f"  [信息] 未指定 --date，自动使用最新交易日: {latest.strftime('%Y-%m-%d')}")
    return latest


# ==================== Step 2: 构造单日候选样本 ====================

def _build_candidates(target_date, engine, position_codes: list = None):
    """
    从 DB 构造 obs_date=T 的单日候选样本。
    复用 01_build_dataset 的函数，但只加载必要的信号和 K 线。
    若提供 position_codes，将持仓股合并到候选池统一计算。
    返回 (df_day, kline_dict, msg)。
    """
    target_dt = pd.to_datetime(target_date) if isinstance(target_date, str) else target_date
    signal_start = (target_dt - pd.Timedelta(days=SIGNAL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    signal_end = target_dt.strftime("%Y-%m-%d")

    # 读信号
    with engine.connect() as conn:
        sql = """
            SELECT id, selection_date, signal_date, ts_code, stock_name,
                   sell_stop_triggered, buy_stop_triggered,
                   sell_trigger_volume, buy_trigger_volume,
                   active_sell_cluster_count, active_buy_cluster_count,
                   sum_sells_active, sum_buys_active,
                   sell_trigger_max_vol_price, sell_stop_scale,
                   nearest_sell_stop_price, nearest_buy_stop_price,
                   dist_to_nearest_sell_stop_atr, dist_to_nearest_buy_stop_atr,
                   stop_cluster_ratio, change_pct, vol_zscore,
                   bbmacd_event, daily_bb_width_zscore,
                   last_event_type, last_event_volume, last_event_bars_ago
            FROM stop_loss_selection
            WHERE signal_date >= :sig_start AND signal_date <= :sig_end
            ORDER BY signal_date, ts_code
        """
        signals = pd.read_sql(text(sql), conn, params={
            "sig_start": signal_start, "sig_end": signal_end,
        })
    print(f"  [信号] {len(signals)} 条 (signal_date {signal_start} ~ {signal_end})")

    if signals.empty:
        return None, None, "当日无信号"

    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    signals["selection_date"] = pd.to_datetime(signals["selection_date"])

    ts_codes = sorted(signals["ts_code"].unique())
    min_signal = signals["signal_date"].min()
    max_signal = signals["signal_date"].max()

    # 读 K 线（扩展范围确保因子 lookback 与训练管线一致）
    # 训练时 batch_kline_start = min(selection_date) - 150天，最早信号约 2023-03
    # 即训练 K 线从约 2022-10 开始。为确保滚动因子有足够 warm-up，
    # 从 target_date - 1300天（约 3.5 年）开始加载，与训练口径对齐
    kline_start = (pd.Timestamp(target_dt) - pd.Timedelta(days=1300)).strftime("%Y-%m-%d")
    kline_end = (pd.Timestamp(max_signal) + pd.Timedelta(days=OBS_DAYS + 50)).strftime("%Y-%m-%d")

    kline_dict = {}
    with engine.connect() as conn:
        for ts_code in ts_codes:
            sql_k = """
                SELECT bar_time, open, high, low, close, volume
                FROM stock_k_data
                WHERE ts_code = :ts_code AND freq = 'd'
                  AND bar_time >= :start_date AND bar_time <= :end_date
                ORDER BY bar_time
            """
            df_k = pd.read_sql(text(sql_k), conn, params={
                "ts_code": ts_code, "start_date": kline_start, "end_date": kline_end,
            })
            if not df_k.empty:
                df_k["bar_time"] = pd.to_datetime(df_k["bar_time"])
                df_k = df_k.set_index("bar_time")
                kline_dict[ts_code] = df_k

    print(f"  [K线] {len(kline_dict)}/{len(ts_codes)} 只 ({kline_start} ~ {kline_end})")

    if not kline_dict:
        return None, None, "K线数据为空"

    # 展开观察期（复用 01 的核心逻辑）
    _mod01 = __import__("stop_experiment.pipeline.01_build_dataset", fromlist=["expand_observation_period"])
    expand_observation_period = _mod01.expand_observation_period
    df_expanded = expand_observation_period(signals, kline_dict)
    print(f"  [展开] {len(df_expanded)} 行 (所有 obs_day)")

    # 筛选 obs_date == T
    df_expanded["obs_date"] = pd.to_datetime(df_expanded["obs_date"])
    df_day = df_expanded[df_expanded["obs_date"] == target_dt].copy()
    print(f"  [筛选] obs_date={target_dt.strftime('%Y-%m-%d')}: {len(df_day)} 行")

    # 合并持仓股到候选池（统一走因子计算路径）
    if position_codes:
        existing_codes = set(df_day["ts_code"].unique()) if not df_day.empty else set()
        if "obs_day" in df_day.columns:
            existing_in_obs = set(df_day[df_day["obs_day"].isin(CANDIDATE_OBS_DAYS)]["ts_code"].unique())
        else:
            existing_in_obs = existing_codes
        new_pos_codes = [c for c in position_codes if c not in existing_in_obs]
        if new_pos_codes:
            print(f"  [持仓合并] {len(new_pos_codes)} 只不在有效候选池(obs_day∈{CANDIDATE_OBS_DAYS})中，加载K线并构造行...")
            pos_rows, pos_kline_dict = _build_position_rows(new_pos_codes, target_date, engine)
            kline_dict.update(pos_kline_dict)
            if not pos_rows.empty:
                for col in ["mfe_20", "mae_20", "sell_signal", "buy_signal"]:
                    if col not in pos_rows.columns:
                        pos_rows[col] = np.nan
                df_day = pd.concat([df_day, pos_rows], ignore_index=True)
                print(f"  [持仓合并] +{len(pos_rows)} 行，共 {len(df_day)} 行")

    if df_day.empty:
        return None, kline_dict, "obs_date=T 无候选"

    return df_day, kline_dict, "OK"


# ==================== Step 3: 因子库 + 派生特征 ====================

def _merge_and_derive(df, kline_dict):
    """拼接因子库特征 → 计算派生特征。复用 01 的函数。"""
    _mod01b = __import__("stop_experiment.pipeline.01_build_dataset", fromlist=["merge_factor_lib_features", "build_derived_features"])
    merge_factor_lib_features = _mod01b.merge_factor_lib_features
    build_derived_features = _mod01b.build_derived_features
    df = merge_factor_lib_features(df, kline_dict)
    df = build_derived_features(df)
    return df


# ==================== Step 4-5: 模型推理 + 打分 ====================

def _predict_and_score(df):
    """加载 final 模型 → 推理 → score_stocks 打分。返回增加了预测列和 score 的 df。"""
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise RuntimeError(f"特征列为空，ALL_FEATURE_COLS 中无列在候选数据中。候选列: {list(df.columns[:20])}...")

    # 确保特征列均为数值类型（LightGBM 要求 int/float/bool）
    for c in feature_cols:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for model_name, spec in MODEL_SPECS.items():
        model_path = os.path.join(MODELS_DIR, f"{model_name}_final.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model = lgb.Booster(model_file=model_path)
        df[f"pred_{model_name}"] = model.predict(df[feature_cols])

    df = score_stocks(df, strategy=PRODUCTION_PARAMS.get("strategy_default", "sell_score"))
    return df


# ==================== Step 6: 写出 ====================

def _save_predictions(df, target_date):
    """写出预测制品到 prediction_store + 数据库。"""
    target_dt = pd.to_datetime(target_date) if isinstance(target_date, str) else target_date
    date_str = target_dt.strftime("%Y-%m-%d")

    if "obs_day" in df.columns:
        before = len(df)
        if "signal_id" in df.columns:
            df_cand = df[df["signal_id"] >= 0]
            df_pos = df[df["signal_id"] < 0]
            df_cand = df_cand[df_cand["obs_day"].isin(CANDIDATE_OBS_DAYS)]
            df = pd.concat([df_cand, df_pos], ignore_index=True)
        else:
            df = df[df["obs_day"].isin(CANDIDATE_OBS_DAYS)]
        if "ts_code" in df.columns and "obs_date" in df.columns:
            df = df.sort_values("obs_day").drop_duplicates(subset=["ts_code", "obs_date"], keep="first")
        if len(df) < before:
            print(f"  [过滤] obs_day∈{CANDIDATE_OBS_DAYS}+去重: {before} → {len(df)} 行")

    available_cols = [c for c in OBS_COLS if c in df.columns]
    df_save = df[available_cols].copy()

    try:
        from stop_experiment.registries.prediction_store import write_prediction_artifact
        store_path = write_prediction_artifact("production", date_str, df_save, prediction_source="live")
        print(f"  [prediction_store] {store_path}")
    except Exception as e:
        print(f"  [prediction_store] 写入失败: {e}")
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        fallback_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.parquet")
        df_save.to_parquet(fallback_path, index=False)
        print(f"  [fallback] {fallback_path} ({len(df_save)} 行, {len(available_cols)} 列)")
        store_path = fallback_path

    try:
        from stop_experiment.registries import load_profile
        profile_data = load_profile("production")
        mv = profile_data.get("model_version", "unknown")
        fv = profile_data.get("feature_version", "unknown")
    except Exception:
        mv = "unknown"
        fv = "unknown"

    _save_predictions_split_profile(df, target_date)

    return store_path


def _infer_missing_position_codes(day_data, target_date, position_codes):
    """对 FTP 提取结果中未覆盖的自选股/持仓股，走完整推理路径补充4模型输出。

    Args:
        day_data: 从 full_test_predictions 提取的当日候选数据
        target_date: 目标交易日
        position_codes: 自选股+持仓股代码列表

    Returns:
        DataFrame: 补充的行（含4模型输出），或 None（无需补充）
    """
    if not position_codes:
        return None

    existing_codes = set(day_data["ts_code"].unique()) if not day_data.empty else set()
    missing_codes = [c for c in position_codes if c not in existing_codes]
    if not missing_codes:
        return None

    print(f"  [补充] {len(missing_codes)} 只自选股/持仓股不在候选池中，走完整推理...")
    engine = get_engine()
    try:
        pos_rows, pos_kline_dict = _build_position_rows(missing_codes, target_date, engine)
        if pos_rows.empty:
            print(f"  [补充] 无K线数据，跳过")
            return None

        df_extra = _merge_and_derive(pos_rows, pos_kline_dict)
        df_extra = _predict_and_score(df_extra)
        print(f"  [补充] 推理完成: {len(df_extra)} 行")
        return df_extra
    except Exception as e:
        print(f"  [补充] 自选股/持仓股推理失败: {e}")
        return None
    finally:
        engine.dispose()


def _save_predictions_split_profile(df, target_date):
    """将预测结果按 signal_id 区分 profile 写入 DB。
    signal_id >= 0 (候选股) → profile='production'
    signal_id < 0 (自选股/持仓股) → profile='position'
    """
    date_str = pd.to_datetime(target_date).strftime("%Y-%m-%d") if not isinstance(target_date, str) else target_date

    try:
        from stop_experiment.registries import load_profile
        profile_data = load_profile("production")
        mv = profile_data.get("model_version", "unknown")
        fv = profile_data.get("feature_version", "unknown")
    except Exception:
        mv = "unknown"
        fv = "unknown"

    from stop_experiment.pipeline.db_writer import save_predictions_to_db

    if "signal_id" in df.columns:
        df_prod = df[df["signal_id"] >= 0].copy()
        df_pos = df[df["signal_id"] < 0].copy()
    else:
        df_prod = df
        df_pos = pd.DataFrame()

    if not df_prod.empty:
        try:
            save_predictions_to_db(df_prod, prediction_date=date_str,
                                   model_version=mv, feature_version=fv,
                                   profile="production")
        except Exception as e:
            print(f"  [DB] production 写入失败: {e}")

    if not df_pos.empty:
        try:
            save_predictions_to_db(df_pos, prediction_date=date_str,
                                   model_version=mv, feature_version=fv,
                                   profile="position")
        except Exception as e:
            print(f"  [DB] position 写入失败: {e}")


# ==================== 主入口 ====================

def generate_daily_predictions(target_date, force=False, position_codes: list = None):
    """6 步日推理主流程。返回 (success, message, output_path)。"""
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    date_str = target_date.strftime("%Y-%m-%d")

    out_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.parquet")
    if os.path.exists(out_path) and not force:
        existing = pd.read_parquet(out_path)
        return True, f"预测账本已存在 ({len(existing)} 行)，跳过。--force 可强制覆盖", out_path

    # 优先从 full_test_predictions 提取（确保与训练口径完全一致）
    ftp_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if os.path.exists(ftp_path) and not force:
        ftp = pd.read_parquet(ftp_path)
        if "obs_date" in ftp.columns:
            ftp["obs_date"] = pd.to_datetime(ftp["obs_date"])
            day_data = ftp[ftp["obs_date"] == target_date].copy()
            if not day_data.empty:
                if "obs_day" in day_data.columns:
                    day_data = day_data[day_data["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
                    if "ts_code" in day_data.columns and "obs_date" in day_data.columns:
                        day_data = day_data.sort_values("obs_day").drop_duplicates(subset=["ts_code", "obs_date"], keep="first")
                # full_test_predictions 不含 stock_name，从 stop_loss_selection 补充
                if "stock_name" not in day_data.columns:
                    try:
                        _eg = get_engine()
                        with _eg.connect() as conn:
                            ts_codes = day_data["ts_code"].tolist()
                            name_sql = text("SELECT DISTINCT ts_code, stock_name FROM stop_loss_selection WHERE ts_code = ANY(:codes)")
                            name_df = pd.read_sql(name_sql, conn, params={"codes": ts_codes})
                            name_map = dict(zip(name_df["ts_code"], name_df["stock_name"]))
                            day_data["stock_name"] = day_data["ts_code"].map(name_map)
                        _eg.dispose()
                    except Exception:
                        day_data["stock_name"] = ""

                # 检查自选股/持仓股是否已在候选池中，未覆盖的走完整推理路径
                extra_rows = _infer_missing_position_codes(day_data, target_date, position_codes)

                if extra_rows is not None and not extra_rows.empty:
                    day_data = pd.concat([day_data, extra_rows], ignore_index=True)
                    print(f"  [补充] 自选股/持仓股补充 {len(extra_rows)} 行，共 {len(day_data)} 行")

                os.makedirs(PREDICTIONS_DIR, exist_ok=True)
                available_cols = [c for c in OBS_COLS if c in day_data.columns]
                day_save = day_data[available_cols].copy()
                day_save.to_parquet(out_path, index=False)

                # 写入 DB（候选股 production + 自选股/持仓股 position）
                _save_predictions_split_profile(day_data, target_date)

                print(f"  [提取] 从 full_test_predictions 提取 {date_str}: {len(day_save)} 行")
                return True, f"从 full_test_predictions 提取 ({len(day_save)} 行)", out_path

    print(f"\n{'='*60}")
    print(f"  07 单日推理: {date_str}")
    if position_codes:
        print(f"  持仓股: {len(position_codes)} 只")
    print(f"{'='*60}")

    engine = get_engine()

    try:
        # Step 2: 构造候选（含持仓股合并）
        print(f"\n[Step 2] 构造单日候选样本...")
        df_day, kline_dict, msg2 = _build_candidates(target_date, engine, position_codes)
        if df_day is None:
            # 无候选 → 生成空预测账本
            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
            empty_df = pd.DataFrame(columns=OBS_COLS)
            empty_df.to_parquet(out_path, index=False)
            print(f"  [信息] {msg2}，生成空预测账本: {out_path}")
            return True, f"无候选 ({msg2})，空账本已生成", out_path

        # Step 3: 因子库 + 派生（复用 Step 2 的 kline_dict，不重复查询 DB）
        print(f"\n[Step 3] 拼接因子库 + 派生特征...")
        df_day = _merge_and_derive(df_day, kline_dict)

        # can_buy 过滤已移除: T+1 交易场景下涨停板次日可买入，保留全部候选

        # Step 4-5: 推理 + 打分
        print(f"\n[Step 4-5] 模型推理 + 打分...")
        df_day = _predict_and_score(df_day)

        # Step 6: 写出
        print(f"\n[Step 6] 写出预测账本...")
        path = _save_predictions(df_day, target_date)

        return True, f"预测账本已生成 ({len(df_day)} 行)", path

    finally:
        engine.dispose()


def _write_day_to_db(day_data, date_str):
    """将单日预测数据写入 DB（从已有数据回补时使用）。"""
    try:
        from stop_experiment.registries import load_profile
        profile_data = load_profile("production")
        mv = profile_data.get("model_version", "unknown")
        fv = profile_data.get("feature_version", "unknown")
    except Exception:
        mv = "unknown"
        fv = "unknown"
    try:
        from stop_experiment.pipeline.db_writer import save_predictions_to_db
        save_predictions_to_db(day_data, prediction_date=date_str,
                               model_version=mv, feature_version=fv,
                               profile="production")
    except Exception as e:
        print(f"  [DB] 写入失败: {e}")


def backfill_predictions(start_date, end_date, force=False):
    """批量回补指定日期范围的预测结果到 parquet + prediction_store + DB。

    数据源优先级（无需构造因子）：
      1. prediction_store 已有 parquet → 直接读取写入 DB
      2. full_test_predictions.parquet → 一次性加载逐日提取
      3. 以上均无 → 逐日走完整推理流程（仅此路径需要构造因子）

    返回 (total, success_count, fail_dates)。
    """
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    engine = get_engine()
    try:
        trading_days = _get_trading_days(sd, ed, engine)
    finally:
        engine.dispose()

    if trading_days.empty:
        print(f"  [回补] {sd.strftime('%Y-%m-%d')} ~ {ed.strftime('%Y-%m-%d')} 无交易日")
        return 0, 0, []

    total = len(trading_days)
    print(f"\n{'='*60}")
    print(f"  回补模式: {sd.strftime('%Y-%m-%d')} ~ {ed.strftime('%Y-%m-%d')}")
    print(f"  交易日: {total} 天")
    print(f"{'='*60}")

    try:
        from stop_experiment.registries.prediction_store import read_prediction_artifact
        has_store = True
    except Exception:
        has_store = False

    ftp_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    ftp = None

    success_count = 0
    fail_dates = []

    for i, td in enumerate(trading_days, 1):
        date_str = td.strftime("%Y-%m-%d")
        print(f"\n--- [{i}/{total}] {date_str} ---")

        try:
            day_data = None

            if has_store and not force:
                artifact = read_prediction_artifact("production", date_str)
                if artifact is not None and not artifact.empty:
                    day_data = artifact
                    print(f"  [prediction_store] 读取 {len(day_data)} 行")

            if day_data is None and not force:
                if ftp is None and os.path.exists(ftp_path):
                    print(f"  [FTP] 加载 full_test_predictions.parquet")
                    ftp = pd.read_parquet(ftp_path)
                    if "obs_date" in ftp.columns:
                        ftp["obs_date"] = pd.to_datetime(ftp["obs_date"])
                if ftp is not None and "obs_date" in ftp.columns:
                    day_data = ftp[ftp["obs_date"] == td].copy()
                    if not day_data.empty:
                        if "obs_day" in day_data.columns:
                            day_data = day_data[day_data["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
                            if "ts_code" in day_data.columns and "obs_date" in day_data.columns:
                                day_data = day_data.sort_values("obs_day").drop_duplicates(subset=["ts_code", "obs_date"], keep="first")
                        print(f"  [FTP] 提取 {len(day_data)} 行")

            if day_data is not None and not day_data.empty:
                os.makedirs(PREDICTIONS_DIR, exist_ok=True)
                available_cols = [c for c in OBS_COLS if c in day_data.columns]
                day_save = day_data[available_cols].copy()
                out_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.parquet")
                day_save.to_parquet(out_path, index=False)
                try:
                    from stop_experiment.registries.prediction_store import write_prediction_artifact
                    write_prediction_artifact("production", date_str, day_save, prediction_source="backfill")
                except Exception as e:
                    print(f"  [prediction_store] 写入失败: {e}")
                _write_day_to_db(day_data, date_str)
                print(f"  ✅ {len(day_save)} 行 → {out_path} + DB")
                success_count += 1
            elif day_data is not None and day_data.empty:
                print(f"  [跳过] 无数据")
                success_count += 1
            else:
                ok, msg, path = generate_daily_predictions(td, force=force)
                if ok:
                    print(f"  ✅ {msg}")
                    success_count += 1
                else:
                    print(f"  ❌ {msg}")
                    fail_dates.append(date_str)
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            fail_dates.append(date_str)

    print(f"\n{'='*60}")
    print(f"  回补完成: {success_count}/{total} 成功")
    if fail_dates:
        print(f"  失败日期: {fail_dates}")
    print(f"{'='*60}")

    return total, success_count, fail_dates


# ==================== 持仓股票行构造 ====================

def _build_position_rows(ts_codes: list, target_date, engine) -> tuple:
    """为持仓股构造 obs_date=T, obs_day=1 的单行 DataFrame（不含因子列）。

    产出与 expand_observation_period 输出相同结构的行，但 SLC 列填 0、
    DYNAMIC 列从当天 K 线计算或填 NaN，以便与候选池 DataFrame 合并后
    统一走 merge_factor_lib_features → build_derived_features 路径。

    Returns:
        (rows_df, kline_dict)：
        - rows_df: DataFrame(ts_code, stock_name, signal_id, obs_date, obs_close, can_buy,
                  SLC_STATIC_COLS, DYNAMIC_COLS)，或空 DataFrame
        - kline_dict: {ts_code: DataFrame} 用于 merge_factor_lib_features
    """
    target_dt = pd.to_datetime(target_date) if isinstance(target_date, str) else target_date
    kline_start = (target_dt - pd.Timedelta(days=1300)).strftime("%Y-%m-%d")
    kline_end = target_dt.strftime("%Y-%m-%d")

    with engine.connect() as conn:
        sql_k = """
            SELECT ts_code, bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE ts_code = ANY(:ts_codes) AND freq = 'd'
              AND bar_time >= :start_date AND bar_time <= :end_date
            ORDER BY ts_code, bar_time
        """
        df_k_all = pd.read_sql(text(sql_k), conn, params={
            "ts_codes": ts_codes, "start_date": kline_start, "end_date": kline_end,
        })

    if df_k_all.empty:
        print(f"    [持仓] 无K线数据")
        return pd.DataFrame(), {}

    df_k_all["bar_time"] = pd.to_datetime(df_k_all["bar_time"])

    rows = []
    kline_dict = {}
    for ts_code in ts_codes:
        df_k = df_k_all[df_k_all["ts_code"] == ts_code].copy()
        if df_k.empty:
            continue
        df_k = df_k.set_index("bar_time").sort_index()
        kline_dict[ts_code] = df_k

        if target_dt not in df_k.index:
            nearest = df_k.index[df_k.index <= target_dt]
            if nearest.empty:
                del kline_dict[ts_code]
                continue
            actual_date = nearest[-1]
        else:
            actual_date = target_dt

        actual_date = pd.Timestamp(actual_date)
        obs_close = df_k.loc[actual_date, "close"]

        row = {"ts_code": ts_code, "signal_id": -1, "obs_date": actual_date,
               "obs_close": obs_close, "stock_name": "", "can_buy": True,
               "sell_stop_triggered": False, "sell_stop_scale": 0.0}

        for col in SLC_STATIC_COLS:
            row[col] = 0

        dynamic = _compute_position_dynamic(actual_date, df_k)
        row.update(dynamic)

        rows.append(row)

    if not rows:
        return pd.DataFrame(), {}

    return pd.DataFrame(rows), kline_dict


def _compute_position_dynamic(obs_date, df_k: pd.DataFrame) -> dict:
    """为持仓股计算 DYNAMIC_COLS 值（与 expand_observation_period 口径对齐）。"""
    obs_close = df_k.loc[obs_date, "close"]
    obs_idx = df_k.index.get_loc(obs_date)

    dynamic = {
        "obs_day": 1,
        "ret_to_trigger": 0.0,
        "high_to_trigger": np.nan,
        "low_to_trigger": np.nan,
        "intraday_range": np.nan,
        "vol_ratio": np.nan,
        "range_position": np.nan,
        "vol_change": np.nan,
        "dist_to_sell_stop_pct": np.nan,
        "dist_to_buy_stop_pct": np.nan,
    }

    if obs_idx < 1 or len(df_k) < 2:
        return dynamic

    prev_idx = obs_idx - 1
    prev_close = df_k.iloc[prev_idx]["close"]

    if obs_close > 0:
        dynamic["intraday_range"] = (
            df_k.loc[obs_date, "high"] - df_k.loc[obs_date, "low"]
        ) / obs_close
        dynamic["ret_to_trigger"] = (obs_close - prev_close) / prev_close if prev_close > 0 else np.nan
        dynamic["high_to_trigger"] = df_k.loc[obs_date, "high"] / prev_close if prev_close > 0 else np.nan
        dynamic["low_to_trigger"] = df_k.loc[obs_date, "low"] / prev_close if prev_close > 0 else np.nan

    vol_start = max(0, obs_idx - 20)
    vol_mean = df_k.iloc[vol_start:obs_idx]["volume"].mean()
    dynamic["vol_ratio"] = df_k.loc[obs_date, "volume"] / vol_mean if vol_mean > 0 else np.nan

    if prev_close > 0 and df_k.iloc[prev_idx]["volume"] > 0:
        dynamic["vol_change"] = df_k.loc[obs_date, "volume"] / df_k.iloc[prev_idx]["volume"] - 1

    return dynamic


def main():
    parser = argparse.ArgumentParser(description="每日推理生成器: 构造单日候选 → 模型推理 → 预测账本")
    parser.add_argument("--date", help="目标交易日 YYYY-MM-DD")
    parser.add_argument("--start-date", help="回补起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", help="回补结束日期 YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的预测账本")
    parser.add_argument("--no-positions", action="store_true", help="禁用持仓股模式")
    parser.add_argument("--no-watchlist", action="store_true", help="禁用自选股模式（默认启用）")
    args = parser.parse_args()

    if args.start_date and args.end_date:
        total, ok, fails = backfill_predictions(args.start_date, args.end_date, force=args.force)
        if fails:
            sys.exit(1)
        return

    engine = get_engine()
    try:
        target_date = _resolve_trade_date(args, engine)
    finally:
        engine.dispose()

    position_codes = None
    if not args.no_positions:
        try:
            from qmt_trader.client import QmtClient
            from qmt_trader.session import SessionManager
            from qmt_trader.query import QueryAPI

            client = QmtClient()
            if not client.health():
                raise ConnectionError("QMT 服务不可达")
            mgr = SessionManager(client)
            sess = mgr.open()
            try:
                api = QueryAPI(client)
                positions = api.get_positions(sess.session_id)
            finally:
                mgr.close(sess.session_id)
            position_codes = [p.stock_code for p in positions]
        except Exception:
            try:
                from qmt_trader.mock import MockQmtClient
                from qmt_trader.query import QueryAPI
                client = MockQmtClient()
                api = QueryAPI(client)
                positions = api.get_positions("mock_session")
                position_codes = [p.stock_code for p in positions]
            except Exception as e:
                print(f"  ⚠ 无法获取持仓: {e}")
                position_codes = None

    watchlist_codes = None
    if not args.no_watchlist:
        try:
            from datasource.database import get_engine as _get_engine
            _eng = _get_engine()
            with _eng.connect() as _conn:
                _result = _conn.execute(text("SELECT ts_code FROM stock_watchlist"))
                watchlist_codes = [row[0] for row in _result]
            _eng.dispose()
            if watchlist_codes:
                print(f"  自选股: {len(watchlist_codes)} 只")
        except Exception as e:
            print(f"  ⚠ 无法获取自选股: {e}")
            watchlist_codes = None

    all_extra_codes = list(set((position_codes or []) + (watchlist_codes or [])))

    success, msg, path = generate_daily_predictions(target_date, force=args.force, position_codes=all_extra_codes)
    print(f"\n{msg}")
    if success:
        print(f"  ✅ 完成: {path}")
    else:
        print(f"  ❌ 失败")
        sys.exit(1)


if __name__ == "__main__":
    main()