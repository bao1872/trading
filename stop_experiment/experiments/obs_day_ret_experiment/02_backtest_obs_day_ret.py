#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obs_day_ret 因子模型重训 + 回测对比实验

Purpose:
    在端到端回测中验证加入 obs_day_ret 后模型和策略的实际改善。
    对照组 A: 当前基线特征集（ALL_FEATURE_COLS，57列）
    实验组 B: 基线 + obs_day_ret（58列）

Inputs:
    - stop_experiment/output/dataset_batches/ 或 dataset.parquet
    - DB: stock_k_data

Outputs:
    - results/backtest/model_metrics_comparison.csv
    - results/backtest/obs_day_ret_importance.csv
    - results/backtest/backtest_comparison.csv
    - output/models_obs_day_ret/ (实验组模型)

How to Run:
    python -m stop_experiment.experiments.obs_day_ret_experiment.02_backtest_obs_day_ret

Side Effects:
    - 只读 DB 和 parquet
    - 输出模型到 output/models_obs_day_ret/
    - 输出结果到 results/backtest/
"""

from __future__ import annotations

import sys
import os
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

from stop_experiment.pipeline.stop_config import (
    EMBARGO_DAYS, LGB_PARAMS, MODEL_SPECS,
    OUTPUT_DIR, DATASET_PATH, MODELS_DIR,
    OBS_TRAIN_END, OBS_VAL_END, OBS_VAL_END,
    BUY_COST, SELL_COST,
)
from stop_experiment.pipeline.factor_columns import ALL_FEATURE_COLS
from stop_experiment.backtest.simple_backtest import (
    load_daily_prices, build_price_pivot,
    is_limit_up, is_limit_down, is_suspended,
)
from stop_experiment.backtest.decision_core import decide_eod

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
BACKTEST_DIR = os.path.join(RESULTS_DIR, "backtest")
BATCHES_DIR = os.path.join(OUTPUT_DIR, "dataset_batches")

MAX_HOLD_DAYS = 20
STOP_LOSS = -0.07
EXIT_THRESHOLD = 0.70
MAX_STOCKS = 10


def compute_obs_day_ret(df: pd.DataFrame) -> pd.Series:
    """计算观察日当天涨幅（与 01 脚本逻辑一致）"""
    df = df.sort_values(["signal_id", "obs_day"]).copy()
    obs_day_ret = df.groupby("signal_id")["obs_close"].pct_change()

    mask_d1 = df["obs_day"] == 1
    ret_to_trigger_safe = df.loc[mask_d1, "ret_to_trigger"].replace(0, np.nan)
    signal_close = df.loc[mask_d1, "obs_close"] / (1 + ret_to_trigger_safe)
    obs_day_ret_d1 = df.loc[mask_d1, "obs_close"] / signal_close - 1
    obs_day_ret.iloc[mask_d1.values.nonzero()[0]] = obs_day_ret_d1.values

    obs_day_ret = obs_day_ret.replace([np.inf, -np.inf], np.nan)
    return obs_day_ret


def load_dataset() -> pd.DataFrame:
    """加载数据集，自动兼容分批/单文件模式"""
    manifest_path = os.path.join(BATCHES_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  分批数据集: {manifest['total_batches']} 批, {manifest['total_rows']} 行")
        dfs = []
        for batch_file in tqdm(manifest["batch_files"], desc="加载分批数据集"):
            batch_path = os.path.join(BATCHES_DIR, batch_file)
            dfs.append(pd.read_parquet(batch_path))
        return pd.concat(dfs, ignore_index=True)
    elif os.path.exists(DATASET_PATH):
        print(f"  单文件数据集: {DATASET_PATH}")
        return pd.read_parquet(DATASET_PATH)
    else:
        raise FileNotFoundError(f"数据集不存在: {BATCHES_DIR} 或 {DATASET_PATH}")


@dataclass
class DataSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_train_val_test_split(df: pd.DataFrame) -> DataSplit:
    """基于 obs_date 的 train/val/test 分割"""
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    obs_train_end_ts = pd.Timestamp(OBS_TRAIN_END)
    obs_val_end_ts = pd.Timestamp(OBS_VAL_END)
    embargo_td = pd.Timedelta(days=EMBARGO_DAYS)

    train_cutoff = obs_train_end_ts - embargo_td
    train_mask = df["obs_date"] <= train_cutoff
    val_mask = (df["obs_date"] > obs_train_end_ts) & (df["obs_date"] <= obs_val_end_ts)
    test_mask = df["obs_date"] > obs_val_end_ts

    return DataSplit(
        train_idx=df.index[train_mask].values,
        val_idx=df.index[val_mask].values,
        test_idx=df.index[test_mask].values,
    )


def train_single_model(df, feature_cols, target_col, train_idx, val_idx,
                       params, task="regression", model_name_tag=""):
    """训练单个 LightGBM 模型"""
    cat_cols = [c for c in ["trend_align_momo", "dsa_dir", "bbmacd_sign",
                             "prev_pivot_code", "price_vol_coord", "has_buy_cluster"]
                if c in feature_cols]

    train_data = lgb.Dataset(
        df.loc[train_idx, feature_cols], df.loc[train_idx, target_col],
        categorical_feature=cat_cols, free_raw_data=False,
    )
    val_data = lgb.Dataset(
        df.loc[val_idx, feature_cols], df.loc[val_idx, target_col],
        reference=train_data, categorical_feature=cat_cols, free_raw_data=False,
    )

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data, num_boost_round=1000,
        valid_sets=[val_data], callbacks=callbacks,
    )

    y_val = df.loc[val_idx, target_col]
    y_val_pred = model.predict(df.loc[val_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_val.notna()
    y_val_v = y_val[valid_mask].values
    y_val_pred_v = y_val_pred[valid_mask]

    val_metrics = {"split": "val", "best_iteration": model.best_iteration}
    if task == "regression":
        val_metrics["eval_mae"] = mean_absolute_error(y_val_v, y_val_pred_v) if len(y_val_v) > 0 else np.nan
        val_metrics["eval_ic"] = stats.spearmanr(y_val_v, y_val_pred_v)[0] if len(y_val_v) > 10 else np.nan
    else:
        if len(np.unique(y_val_v)) > 1 and len(y_val_v) > 0:
            val_metrics["eval_auc"] = roc_auc_score(y_val_v, y_val_pred_v)
            val_metrics["eval_ap"] = average_precision_score(y_val_v, y_val_pred_v)
        else:
            val_metrics["eval_auc"] = np.nan
            val_metrics["eval_ap"] = np.nan

    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_cols, "gain": importance, "model_name": model_name_tag,
    })

    return model, val_metrics, imp_df


def evaluate_on_test(model, df, feature_cols, target_col, test_idx, task, model_name):
    """在测试集上评估模型"""
    y_test = df.loc[test_idx, target_col]
    y_pred = model.predict(df.loc[test_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_test.notna()
    y_test_v = y_test[valid_mask].values
    y_pred_v = y_pred[valid_mask]

    test_metrics = {"model_name": model_name, "split": "test"}
    if task == "regression":
        test_metrics["test_mae"] = mean_absolute_error(y_test_v, y_pred_v) if len(y_test_v) > 0 else np.nan
        test_metrics["test_ic"] = stats.spearmanr(y_test_v, y_pred_v)[0] if len(y_test_v) > 10 else np.nan
    else:
        if len(np.unique(y_test_v)) > 1 and len(y_test_v) > 0:
            test_metrics["test_auc"] = roc_auc_score(y_test_v, y_pred_v)
            test_metrics["test_ap"] = average_precision_score(y_test_v, y_pred_v)
        else:
            test_metrics["test_auc"] = np.nan
            test_metrics["test_ap"] = np.nan
    return test_metrics


def train_all_models(df, feature_cols, split, output_dir, tag=""):
    """训练 4 个模型变体，返回模型字典和指标"""
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    all_importance = []
    models = {}
    best_iterations = {}

    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        objective = spec["objective"]
        metric_name = spec["metric"]
        task = "classification" if objective == "binary" else "regression"

        print(f"\n  --- {model_name} ({tag}) ---")
        params = LGB_PARAMS.copy()
        params["objective"] = objective
        params["metric"] = metric_name

        if task == "classification":
            pos_rate = df.loc[split.train_idx, target].mean()
            if pos_rate > 0:
                params["scale_pos_weight"] = (1 - pos_rate) / pos_rate

        model, val_metrics, imp_df = train_single_model(
            df, feature_cols, target,
            split.train_idx, split.val_idx,
            params, task, model_name_tag=f"{model_name}_{tag}",
        )
        val_metrics["model_name"] = model_name
        val_metrics["tag"] = tag
        all_metrics.append(val_metrics)
        all_importance.append(imp_df)
        models[model_name] = model
        best_iterations[model_name] = model.best_iteration

        test_metrics = evaluate_on_test(
            model, df, feature_cols, target,
            split.test_idx, task, model_name,
        )
        test_metrics["tag"] = tag
        all_metrics.append(test_metrics)

        if task == "regression":
            print(f"    val: MAE={val_metrics.get('eval_mae', np.nan):.4f}, "
                  f"IC={val_metrics.get('eval_ic', np.nan):.4f}")
            print(f"    test: MAE={test_metrics.get('test_mae', np.nan):.4f}, "
                  f"IC={test_metrics.get('test_ic', np.nan):.4f}")
        else:
            print(f"    val: AUC={val_metrics.get('eval_auc', np.nan):.4f}, "
                  f"AP={val_metrics.get('eval_ap', np.nan):.4f}")
            print(f"    test: AUC={test_metrics.get('test_auc', np.nan):.4f}, "
                  f"AP={test_metrics.get('test_ap', np.nan):.4f}")

        model_path = os.path.join(output_dir, f"{model_name}.txt")
        model.save_model(model_path)

    train_val_idx = np.concatenate([split.train_idx, split.val_idx])
    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        objective = spec["objective"]
        task = "classification" if objective == "binary" else "regression"

        params = LGB_PARAMS.copy()
        params["objective"] = objective
        params["metric"] = spec["metric"]

        if task == "classification":
            pos_rate = df.loc[train_val_idx, target].mean()
            if pos_rate > 0:
                params["scale_pos_weight"] = (1 - pos_rate) / pos_rate

        cat_cols = [c for c in ["trend_align_momo", "dsa_dir", "bbmacd_sign",
                                 "prev_pivot_code", "price_vol_coord", "has_buy_cluster"]
                    if c in feature_cols]
        train_data = lgb.Dataset(
            df.loc[train_val_idx, feature_cols], df.loc[train_val_idx, target],
            categorical_feature=cat_cols, free_raw_data=False,
        )
        final_params = params.copy()
        final_params["verbosity"] = -1
        final_model = lgb.train(final_params, train_data, num_boost_round=best_iterations[model_name])

        final_model_path = os.path.join(output_dir, f"{model_name}_final.txt")
        final_model.save_model(final_model_path)
        print(f"    最终模型: {final_model_path} (iter={best_iterations[model_name]})")

    return models, pd.DataFrame(all_metrics), pd.concat(all_importance, ignore_index=True)


def build_pred_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in df.iterrows():
        pred_dict = {
            "pred_buy_cls": float(row.get("pred_buy_cls", np.nan)),
            "pred_sell_reg": float(row.get("pred_sell_reg", np.nan)),
            "pred_sell_cls": float(row.get("pred_sell_cls", np.nan)),
            "pred_buy_reg": float(row.get("pred_buy_reg", np.nan)),
        }
        sid_key = (int(row["signal_id"]), row["obs_date"])
        lookup[sid_key] = pred_dict
        ts_code = row.get("ts_code")
        if ts_code:
            ts_key = (ts_code, row["obs_date"])
            lookup[ts_key] = pred_dict
    return lookup


def compute_summary(result: dict) -> dict:
    nav_df = result["nav_df"].copy()
    trades_df = result["trades_df"]
    if nav_df.empty:
        return {"n_trades": 0, "final_nav": 1.0, "annual_ret": 0, "sharpe": 0,
                "max_dd": 0, "win_rate": 0, "avg_net_ret": 0, "avg_hold_days": 0}
    nav_df["cummax"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = (nav_df["nav"] - nav_df["cummax"]) / nav_df["cummax"]
    total_days = len(nav_df)
    total_years = total_days / 252
    final_nav = nav_df["nav"].iloc[-1]
    annual_ret = (final_nav ** (1 / total_years) - 1) if total_years > 0 and final_nav > 0 else 0
    max_dd = nav_df["drawdown"].min()
    daily_rets = nav_df["daily_ret"]
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 1e-6 else 0
    n_trades = len(trades_df)
    win_rate = (trades_df["net_ret"] > 0).mean() if n_trades > 0 else 0
    avg_net_ret = trades_df["net_ret"].mean() if n_trades > 0 else 0
    avg_hold = trades_df["hold_days"].mean() if n_trades > 0 else 0
    profit_loss_ratio = (trades_df[trades_df["net_ret"] > 0]["net_ret"].mean() /
                         abs(trades_df[trades_df["net_ret"] < 0]["net_ret"].mean())
                         if (trades_df["net_ret"] > 0).any() and (trades_df["net_ret"] < 0).any() else 0)
    return {"n_trades": n_trades, "final_nav": final_nav, "annual_ret": annual_ret,
            "max_dd": max_dd, "sharpe": sharpe, "win_rate": win_rate,
            "avg_net_ret": avg_net_ret, "avg_hold_days": avg_hold,
            "profit_loss_ratio": profit_loss_ratio}


def run_backtest(signals_df, price_pivot, trading_days, prev_close_map, pred_lookup):
    signals_sorted = signals_df.sort_values(["obs_date", "pred_sell_reg"], ascending=[True, False])
    signal_dates = sorted(signals_df["obs_date"].unique())
    signal_by_date = {}
    for date in signal_dates:
        day_sigs = signals_sorted[signals_sorted["obs_date"] == date]
        signal_by_date[date] = day_sigs.drop_duplicates(subset=["ts_code"], keep="first")

    holdings = {}
    pending_orders = []
    pending_sells = []
    trade_details = []
    nav_records = []
    skipped = defaultdict(int)
    empty_pool_days = 0

    for t_idx, current_date in enumerate(trading_days):
        if current_date not in price_pivot.index:
            continue

        day_open = price_pivot.loc[current_date, "open"] if "open" in price_pivot else pd.Series(dtype=float)
        day_close = price_pivot.loc[current_date, "close"] if "close" in price_pivot else pd.Series(dtype=float)

        if pending_sells:
            for sell_item in pending_sells:
                code = sell_item["code"]
                h = sell_item["holding"]
                sell_price = np.nan
                if code in day_open.index and not np.isnan(day_open[code]):
                    sell_price = day_open[code]
                if np.isnan(sell_price) or sell_price <= 0:
                    skipped["no_sell_price"] += 1
                    continue
                if "volume" in price_pivot and code in price_pivot["volume"].columns:
                    vol_c = price_pivot["volume"][code].get(current_date, np.nan)
                    if is_suspended(vol_c):
                        skipped["suspended_sell"] += 1
                        continue
                if code in prev_close_map and current_date in prev_close_map[code].index:
                    prev_c = prev_close_map[code].get(current_date, np.nan)
                    if not np.isnan(prev_c) and prev_c > 0:
                        if "low" in price_pivot:
                            dl = price_pivot["low"][code]
                            if current_date in dl.index and not np.isnan(dl[current_date]):
                                if is_limit_down(sell_price, dl[current_date], prev_c):
                                    skipped["limit_down"] += 1
                                    continue
                gross_ret = (sell_price - h["buy_price"]) / h["buy_price"]
                net_ret = gross_ret - BUY_COST - SELL_COST
                trade_details.append({
                    "ts_code": h["ts_code"], "buy_date": h["buy_date"],
                    "sell_date": current_date, "buy_price": h["buy_price"],
                    "sell_price": sell_price, "hold_days": h["days_held"],
                    "gross_ret": gross_ret, "net_ret": net_ret,
                    "sell_reason": sell_item["reason"], "score": h.get("score", 0),
                })
                if code in holdings:
                    del holdings[code]
            pending_sells = []

        _buy_max = MAX_STOCKS - len(holdings)
        if _buy_max < 0:
            _buy_max = 0
        if pending_orders:
            executed = []
            for code, bp, ts_code, sc, sid in pending_orders:
                if len(executed) >= _buy_max:
                    break
                if code in holdings:
                    skipped["already_held"] += 1
                    continue
                if "volume" in price_pivot and code in price_pivot["volume"].columns:
                    vol_c = price_pivot["volume"][code].get(current_date, np.nan)
                    if is_suspended(vol_c):
                        skipped["suspended"] += 1
                        continue
                if code in prev_close_map and current_date in prev_close_map[code].index:
                    prev_c = prev_close_map[code].get(current_date, np.nan)
                    if not np.isnan(prev_c) and prev_c > 0:
                        if "high" in price_pivot:
                            dh = price_pivot["high"][code]
                            if current_date in dh.index:
                                if is_limit_up(bp, dh.get(current_date, np.nan), prev_c):
                                    skipped["limit_up"] += 1
                                    continue
                executed.append((code, bp, ts_code, sc, sid))
            if executed:
                n = len(holdings) + len(executed)
                w = 1.0 / n
                for code_h in holdings:
                    holdings[code_h]["weight"] = w
                for code, bp, ts_code, sc, sid in executed:
                    holdings[code] = {
                        "buy_date": current_date, "buy_price": bp,
                        "weight": w, "days_held": 0,
                        "ts_code": ts_code, "score": sc, "signal_id": sid,
                    }
            pending_orders = []

        prev_date = trading_days[t_idx - 1] if t_idx > 0 else None
        next_idx = t_idx + 1
        day_open_next = (price_pivot.loc[trading_days[next_idx], "open"]
                         if next_idx < len(trading_days) else pd.Series(dtype=float))

        candidates = signal_by_date.get(current_date, pd.DataFrame())
        if candidates.empty:
            empty_pool_days += 1

        holdings, pending_buys_new, pending_sells_new, sell_reasons, _ = decide_eod(
            decision_date=current_date,
            holdings=holdings,
            candidates=candidates,
            pred_lookup=pred_lookup,
            prev_date=prev_date,
            day_close=day_close,
            day_open_next=day_open_next,
            max_stocks=MAX_STOCKS,
            max_hold_days=MAX_HOLD_DAYS,
            stop_loss=STOP_LOSS,
            exit_threshold=EXIT_THRESHOLD,
        )

        pending_sells = pending_sells_new
        pending_orders = pending_buys_new

        daily_ret = 0.0
        for code, h in holdings.items():
            if code in day_close.index and not np.isnan(day_close[code]):
                if h["days_held"] == 1:
                    if code in day_open.index and not np.isnan(day_open[code]):
                        sr = (day_close[code] - day_open[code]) / day_open[code]
                    else:
                        sr = 0
                elif t_idx > 0:
                    prev_d = trading_days[t_idx - 1]
                    if prev_d in price_pivot.index:
                        prev_c = price_pivot.loc[prev_d, "close"]
                        if code in prev_c.index and not np.isnan(prev_c[code]):
                            sr = (day_close[code] - prev_c[code]) / prev_c[code]
                        else:
                            sr = 0
                    else:
                        sr = 0
                else:
                    sr = 0
                daily_ret += h["weight"] * sr

        prev_nav = nav_records[-1]["nav"] if nav_records else 1.0
        nav = prev_nav * (1 + daily_ret)
        nav_records.append({"date": current_date, "nav": nav, "daily_ret": daily_ret,
                            "n_positions": len(holdings)})

    nav_df = pd.DataFrame(nav_records)
    trades_df = pd.DataFrame(trade_details)
    return {"nav_df": nav_df, "trades_df": trades_df,
            "skipped_stats": dict(skipped), "empty_pool_days": empty_pool_days}


def main():
    print("=" * 60)
    print("obs_day_ret 因子模型重训 + 回测对比实验")
    print("=" * 60)

    os.makedirs(BACKTEST_DIR, exist_ok=True)

    print("\n[1/4] 加载数据 + 计算 obs_day_ret...")
    df = load_dataset()
    df = df.dropna(subset=["mfe_20", "mae_20"])
    df = df.sort_values("obs_date").reset_index(drop=True)
    print(f"  有效标签: {len(df)} 行")

    df["obs_day_ret"] = compute_obs_day_ret(df)
    valid_ret = df["obs_day_ret"].notna().sum()
    print(f"  obs_day_ret 有效: {valid_ret} ({valid_ret/len(df):.1%})")

    df["obs_day_ret"] = df["obs_day_ret"].fillna(0)

    print("\n[2/4] 构建 train/val/test 分割...")
    split = build_train_val_test_split(df)
    print(f"  train: {len(split.train_idx)}, val: {len(split.val_idx)}, test: {len(split.test_idx)}")

    baseline_features = [c for c in ALL_FEATURE_COLS if c in df.columns and not c.startswith("vsa_")]
    experiment_features = baseline_features + ["obs_day_ret"]
    print(f"  基线特征: {len(baseline_features)}, 实验特征: {len(experiment_features)}")

    print("\n[3/4] 训练模型...")
    print("\n  === A: 基线（不含 obs_day_ret）===")
    baseline_dir = os.path.join(OUTPUT_DIR, "models_obs_day_ret_baseline")
    baseline_models, baseline_metrics, baseline_imp = train_all_models(
        df, baseline_features, split, baseline_dir, tag="baseline",
    )

    print("\n  === B: 实验组（含 obs_day_ret）===")
    experiment_dir = os.path.join(OUTPUT_DIR, "models_obs_day_ret")
    experiment_models, experiment_metrics, experiment_imp = train_all_models(
        df, experiment_features, split, experiment_dir, tag="with_obs_day_ret",
    )

    all_metrics = pd.concat([baseline_metrics, experiment_metrics], ignore_index=True)
    metrics_path = os.path.join(BACKTEST_DIR, "model_metrics_comparison.csv")
    all_metrics.to_csv(metrics_path, index=False)
    print(f"\n  保存指标: {metrics_path}")

    obs_day_ret_imp = experiment_imp[experiment_imp["feature"] == "obs_day_ret"]
    imp_path = os.path.join(BACKTEST_DIR, "obs_day_ret_importance.csv")
    obs_day_ret_imp.to_csv(imp_path, index=False)
    print(f"  保存 obs_day_ret 重要性: {imp_path}")

    print(f"\n  obs_day_ret 因子重要性:")
    for _, row in obs_day_ret_imp.iterrows():
        total_gain = experiment_imp[experiment_imp["model_name"] == row["model_name"]]["gain"].sum()
        pct = row["gain"] / total_gain * 100 if total_gain > 0 else 0
        print(f"    {row['model_name']}: gain={row['gain']:.1f}, 占比={pct:.2f}%")

    print(f"\n  模型指标对比:")
    print(f"  {'模型':12s} {'指标':12s} {'基线':>10s} {'实验':>10s} {'Δ':>10s}")
    for model_name in MODEL_SPECS:
        spec = MODEL_SPECS[model_name]
        task = "classification" if spec["objective"] == "binary" else "regression"
        b_val = baseline_metrics[(baseline_metrics["model_name"] == model_name) & (baseline_metrics["split"] == "test")]
        e_val = experiment_metrics[(experiment_metrics["model_name"] == model_name) & (experiment_metrics["split"] == "test")]
        if len(b_val) == 0 or len(e_val) == 0:
            continue
        if task == "regression":
            for metric_col, metric_label in [("test_mae", "MAE"), ("test_ic", "IC")]:
                bv = b_val[metric_col].iloc[0] if metric_col in b_val.columns else np.nan
                ev = e_val[metric_col].iloc[0] if metric_col in e_val.columns else np.nan
                delta = ev - bv if pd.notna(ev) and pd.notna(bv) else np.nan
                print(f"  {model_name:12s} {metric_label:12s} {bv:>10.4f} {ev:>10.4f} {delta:>+10.4f}")
        else:
            for metric_col, metric_label in [("test_auc", "AUC"), ("test_ap", "AP")]:
                bv = b_val[metric_col].iloc[0] if metric_col in b_val.columns else np.nan
                ev = e_val[metric_col].iloc[0] if metric_col in e_val.columns else np.nan
                delta = ev - bv if pd.notna(ev) and pd.notna(bv) else np.nan
                print(f"  {model_name:12s} {metric_label:12s} {bv:>10.4f} {ev:>10.4f} {delta:>+10.4f}")

    print("\n[4/4] 回测对比...")
    val_end = pd.Timestamp(OBS_VAL_END)
    test_df = df[df["obs_date"] > val_end].copy()
    test_entry = test_df[test_df["obs_day"].isin([1, 2, 3])].copy()
    print(f"  回测样本 (obs_day 1~3): {len(test_entry)}")

    signal_start = str(test_entry["obs_date"].min().date())
    signal_end_dt = test_entry["obs_date"].max() + pd.Timedelta(days=60)
    signal_end = str(signal_end_dt.date())

    print(f"  加载K线: {signal_start} ~ {signal_end}")
    daily_prices = load_daily_prices(signal_start, signal_end)
    price_pivot, trading_days, prev_close_map = build_price_pivot(daily_prices)
    print(f"  交易日: {len(trading_days)}")

    results_rows = []

    for label, models_dir, features in [
        ("A_baseline", baseline_dir, baseline_features),
        ("B_with_obs_day_ret", experiment_dir, experiment_features),
    ]:
        print(f"\n  --- {label} ---")

        pred_df = test_entry.copy()
        for model_name in MODEL_SPECS:
            final_model_path = os.path.join(models_dir, f"{model_name}_final.txt")
            if not os.path.exists(final_model_path):
                final_model_path = os.path.join(models_dir, f"{model_name}.txt")
            model = lgb.Booster(model_file=final_model_path)
            available_features = [c for c in features if c in pred_df.columns]
            pred_df[f"pred_{model_name}"] = model.predict(pred_df[available_features])

        pred_lookup = build_pred_lookup(pred_df)

        result = run_backtest(pred_df, price_pivot, trading_days, prev_close_map, pred_lookup)
        s = compute_summary(result)

        row = {"label": label, **s, "empty_pool_days": result.get("empty_pool_days", 0)}
        results_rows.append(row)

        print(f"    NAV={s['final_nav']:.4f}, Sharpe={s['sharpe']:.2f}, "
              f"MDD={s['max_dd']:.4f}, 胜率={s['win_rate']:.2%}, "
              f"盈亏比={s['profit_loss_ratio']:.2f}, 交易数={s['n_trades']}")

    comp_df = pd.DataFrame(results_rows)
    comp_path = os.path.join(BACKTEST_DIR, "backtest_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  保存: {comp_path}")

    print(f"\n{'='*80}")
    print("回测对比汇总")
    print(f"{'='*80}")
    print(f"  {'方案':25s} {'NAV':>8s} {'Sharpe':>8s} {'MDD':>8s} {'胜率':>8s} {'盈亏比':>8s} {'交易':>5s}")
    for row in results_rows:
        print(f"  {row['label']:25s} {row['final_nav']:>8.4f} {row['sharpe']:>8.2f} "
              f"{row['max_dd']:>8.4f} {row['win_rate']:>8.2%} {row['profit_loss_ratio']:>8.2f} "
              f"{row['n_trades']:>5d}")

    if len(results_rows) >= 2:
        baseline_nav = results_rows[0]["final_nav"]
        experiment_nav = results_rows[1]["final_nav"]
        delta = experiment_nav - baseline_nav
        delta_pct = delta / baseline_nav * 100 if baseline_nav > 0 else 0
        print(f"\n  ΔNAV = {delta:+.4f} ({delta_pct:+.1f}%)")

        if delta > 0 and delta_pct > 2:
            print(f"  ✅ obs_day_ret 提升回测表现 > 2%，建议加入特征集")
        elif delta > 0:
            print(f"  ⚠ obs_day_ret 提升回测表现 < 2%，提升有限")
        else:
            print(f"  ❌ obs_day_ret 未提升回测表现，不建议加入")


if __name__ == "__main__":
    main()
