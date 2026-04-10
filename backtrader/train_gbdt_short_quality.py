# -*- coding: utf-8 -*-
"""
GBDT 训练与验证脚本（3–5天买点质量，v6 rolling + 开关版）

Purpose
- 基于 model_table_short_quality.parquet 训练 GBDT 基线模型。
- 同时支持：
  1) 回归：预测 quality_score_short
  2) 分类：预测 high_quality_flag_short
- 严格按时间切分，禁止随机切分。
- 输出训练集审计、特征统计、模型指标、预测诊断、decile 分组结果、特征重要性。

How to Run
    python train_gbdt_short_quality_v6_toggle_rolling.py \
        --model-table daily_bottom_div_quality_500stocks_final_model_table_short_quality.parquet \
        --out-dir gbdt_short_quality_output_v6

Dependencies
    pip install pandas numpy scikit-learn lightgbm openpyxl pyarrow
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class SplitData:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    name: str


REG_DEFAULT_PARAMS: Dict[str, object] = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "seed": 42,
}

CLF_DEFAULT_PARAMS: Dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "seed": 42,
}

LABEL_EXACT_EXCLUDE = {
    "reward_ret_raw_short",
    "penalty_dd_raw_short",
    "reward_rr_raw_short",
    "quality_hard_penalty_short",
    "quality_raw_short",
    "quality_score_short",
    "quality_bucket_short",
    "high_quality_flag_short",
    "is_quality_score_available",
}
ID_EXCLUDE = {"event_id", "structure_id", "symbol"}
TIME_EXCLUDE = {"trigger_dt", "trigger_year", "trigger_month", "trigger_ym", "trigger_weekday"}
PREFIX_EXCLUDE = ("ret_", "mfe_", "mae_", "rr_")
SUFFIX_EXCLUDE = ("_label", "_target")
EXACT_STRUCT_POSITION_EXCLUDE = {
    # 明显的样本位置/拥挤度代理，默认剔除
    "bar_index",
    "event_count_same_day",
    "symbol_event_rank_same_day",
    "pivot_bar",
    # 结构内排序类元数据：当前样本几乎都等于1，信息量不足且容易引入伪规律
    "event_rank_in_structure",
    "events_in_structure",
    "is_first_event_in_structure",
    "is_last_event_in_structure",
    "bars_since_prev_event_in_structure",
    # 缺失性元数据：先不让模型学“样本缺失形状”
    "feature_missing_count",
    "has_missing_key_feature_flag",
    # 绝对价格水平：跨股票不可比，优先剔除
    "L1_px",
    "H1_px",
    "L2_px",
    "pivot_price",
    "trigger_px",
    "entry_price",
    "trigger_signal",
    # bars 长度类原始字段：先整体剔除一版做对照
    "pb_bars",
    # 强摘要 flag：先拿掉，避免把人工判断硬塞进模型
    "up_trend_stable_flag",
}
CONTAINS_EXCLUDE = (
    "_idx",       # L1_idx/H1_idx/L2_idx/event_idx/entry_idx 这类位置索引
    "index",      # 各种位置类代理
)
PREFIX_TOKEN_EXCLUDE = (
    "bars_from_", # 其他 bars_from_* 原始长度字段
)
SUFFIX_TOKEN_EXCLUDE = (
    "_bar",       # 其他 *bar 位置字段
)
CONTAINS_TOKEN_EXCLUDE = (
    "signal_mean",
    "hist_max",
)


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的输入格式: {path}")


def is_excluded_feature(col: str) -> bool:
    if col in LABEL_EXACT_EXCLUDE or col in ID_EXCLUDE or col in TIME_EXCLUDE or col in EXACT_STRUCT_POSITION_EXCLUDE:
        return True
    if any(col.startswith(p) for p in PREFIX_EXCLUDE):
        return True
    if any(col.endswith(s) for s in SUFFIX_EXCLUDE):
        return True
    lc = col.lower()
    if any(token in lc for token in CONTAINS_EXCLUDE):
        return True
    if any(lc.startswith(token) for token in PREFIX_TOKEN_EXCLUDE):
        return True
    if any(lc.endswith(token) for token in SUFFIX_TOKEN_EXCLUDE):
        return True
    if any(token in lc for token in CONTAINS_TOKEN_EXCLUDE):
        return True
    return False


def is_optionally_excluded_feature(col: str, exclude_up_bars: bool = False, exclude_trigger_macd: bool = False) -> tuple[bool, str]:
    if exclude_up_bars and col == "up_bars":
        return True, "optional_toggle_exclude_up_bars"
    if exclude_trigger_macd and col == "trigger_macd":
        return True, "optional_toggle_exclude_trigger_macd"
    return False, ""



def build_feature_decision_table(df: pd.DataFrame, exclude_up_bars: bool = False, exclude_trigger_macd: bool = False) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=[np.number, bool]).columns:
        decision = "keep_candidate"
        reason = ""
        if col in LABEL_EXACT_EXCLUDE:
            decision, reason = "drop", "label_or_label_derived"
        elif col in ID_EXCLUDE:
            decision, reason = "drop", "id_key"
        elif col in TIME_EXCLUDE:
            decision, reason = "drop", "time_key"
        elif any(col.startswith(p) for p in PREFIX_EXCLUDE):
            decision, reason = "drop", "future_label_window"
        elif any(col.endswith(s) for s in SUFFIX_EXCLUDE):
            decision, reason = "drop", "label_suffix"
        else:
            opt_drop, opt_reason = is_optionally_excluded_feature(col, exclude_up_bars=exclude_up_bars, exclude_trigger_macd=exclude_trigger_macd)
            lc = col.lower()
            if opt_drop:
                decision, reason = "drop", opt_reason
            elif any(token in lc for token in CONTAINS_EXCLUDE):
                decision, reason = "drop", "index_or_position_proxy"
            elif any(lc.startswith(token) for token in PREFIX_TOKEN_EXCLUDE) or any(lc.endswith(token) for token in SUFFIX_TOKEN_EXCLUDE):
                decision, reason = "drop", "bar_length_or_position_proxy"
            elif any(token in lc for token in CONTAINS_TOKEN_EXCLUDE):
                decision, reason = "drop", "raw_macd_level_like"
            elif col in EXACT_STRUCT_POSITION_EXCLUDE:
                if col in {"L1_px", "H1_px", "L2_px", "pivot_price", "trigger_px", "entry_price"}:
                    decision, reason = "drop", "absolute_price_level"
                elif col in {"feature_missing_count", "has_missing_key_feature_flag"}:
                    decision, reason = "drop", "missingness_meta"
                elif col in {"event_rank_in_structure", "events_in_structure", "is_first_event_in_structure", "is_last_event_in_structure", "bars_since_prev_event_in_structure"}:
                    decision, reason = "drop", "weak_structure_meta"
                elif col in {"trigger_signal"}:
                    decision, reason = "drop", "raw_signal_level"
                elif col in {"pb_bars", "up_trend_stable_flag"}:
                    decision, reason = "drop", "bar_length_or_manual_summary"
                else:
                    decision, reason = "drop", "sample_position_or_crowding_proxy"
        rows.append({"feature": col, "decision": decision, "reason": reason})
    return pd.DataFrame(rows).sort_values(["decision", "reason", "feature"]).reset_index(drop=True)

def build_feature_stats(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n = len(df)
    for col in feature_cols:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "missing_count": int(s.isna().sum()),
                "missing_ratio": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
                "is_constant": bool(s.nunique(dropna=True) <= 1),
                "sample_size": n,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_ratio", "feature"], ascending=[False, True]).reset_index(drop=True)


def infer_feature_columns(df: pd.DataFrame, max_missing_ratio: float = 0.60, exclude_up_bars: bool = False, exclude_trigger_macd: bool = False) -> Tuple[List[str], pd.DataFrame]:
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    candidate_cols = [c for c in numeric_cols if not is_excluded_feature(c) and not is_optionally_excluded_feature(c, exclude_up_bars=exclude_up_bars, exclude_trigger_macd=exclude_trigger_macd)[0]]
    stats = build_feature_stats(df, candidate_cols)
    usable = stats[
        (stats["missing_ratio"] <= max_missing_ratio)
        & (~stats["is_constant"])
        & (stats["nunique"] > 1)
    ]["feature"].tolist()
    return usable, stats


def refine_feature_columns_for_source(df: pd.DataFrame, feature_cols: Sequence[str], max_missing_ratio: float = 0.95) -> Tuple[List[str], pd.DataFrame]:
    stats = build_feature_stats(df, feature_cols)
    usable = stats[
        (stats["missing_ratio"] <= max_missing_ratio)
        & (~stats["is_constant"])
        & (stats["nunique"] > 1)
    ]["feature"].tolist()
    return usable, stats


def prepare_targets(df: pd.DataFrame, unify_clf_with_score: bool = True) -> pd.DataFrame:
    out = df.copy()
    if "trigger_dt" not in out.columns:
        raise KeyError("训练表缺少 trigger_dt")
    out["trigger_dt"] = pd.to_datetime(out["trigger_dt"], errors="coerce")
    out["is_quality_score_available"] = out["quality_score_short"].notna().astype(int)
    if unify_clf_with_score:
        mask = out["quality_score_short"].notna()
        out.loc[~mask, "high_quality_flag_short"] = np.nan
    return out


def build_trainset_summary(df: pd.DataFrame, feature_stats: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = [
        ("total_rows", len(df)),
        ("quality_score_available_rows", int(df["quality_score_short"].notna().sum())),
        ("high_quality_flag_available_rows", int(df["high_quality_flag_short"].notna().sum())),
        ("high_quality_positive_rows", int((df["high_quality_flag_short"] == 1).sum())),
        ("high_quality_positive_ratio", float((df["high_quality_flag_short"] == 1).mean() if len(df) else np.nan)),
        ("candidate_feature_count", len(feature_cols)),
        ("raw_numeric_feature_count", int(len(feature_stats))),
        ("features_missing_gt_60pct", int((feature_stats["missing_ratio"] > 0.60).sum())),
        ("constant_features", int(feature_stats["is_constant"].sum())),
        ("date_min", str(df["trigger_dt"].min())),
        ("date_max", str(df["trigger_dt"].max())),
        ("n_symbols", int(df["symbol"].nunique() if "symbol" in df.columns else np.nan)),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def build_time_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df.groupby(df["trigger_dt"].dt.year).size().reset_index(name="count").rename(columns={"trigger_dt": "year"})
    ym = df.groupby(df["trigger_dt"].dt.to_period("M").astype(str)).size().reset_index(name="count").rename(columns={"trigger_dt": "ym"})
    return y, ym


def build_single_time_split(df: pd.DataFrame, train_ratio: float, valid_ratio: float) -> SplitData:
    d = df.sort_values("trigger_dt").reset_index(drop=True)
    n = len(d)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return SplitData(
        train_idx=d.index[:train_end].to_numpy(),
        valid_idx=d.index[train_end:valid_end].to_numpy(),
        test_idx=d.index[valid_end:].to_numpy(),
        name="single_split",
    )


def build_rolling_splits(df: pd.DataFrame, n_folds: int) -> List[SplitData]:
    d = df.sort_values("trigger_dt").reset_index(drop=True)
    n = len(d)
    if n_folds < 1:
        raise ValueError("n_folds 必须 >= 1")
    base = n // (n_folds + 5)
    if base <= 100:
        raise ValueError("样本量过小，无法构建 rolling splits")
    splits: List[SplitData] = []
    for i in range(n_folds):
        train_end = base * (3 + i)
        valid_end = base * (4 + i)
        test_end = base * (5 + i)
        if test_end > n:
            break
        splits.append(SplitData(
            train_idx=d.index[:train_end].to_numpy(),
            valid_idx=d.index[train_end:valid_end].to_numpy(),
            test_idx=d.index[valid_end:test_end].to_numpy(),
            name=f"rolling_fold_{i+1}",
        ))
    if not splits:
        raise ValueError("未能构建有效 rolling split，请减少 folds 或增加样本")
    return splits


def require_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as exc:
        raise RuntimeError("缺少 lightgbm。请先安装: pip install lightgbm") from exc
    return lgb


def fit_lightgbm_regression(X_train, y_train, X_valid, y_valid, params, num_boost_round, early_stopping_rounds):
    lgb = require_lightgbm()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    return lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def fit_lightgbm_classification(X_train, y_train, X_valid, y_valid, params, num_boost_round, early_stopping_rounds):
    lgb = require_lightgbm()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    return lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    s = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return float("nan")
    if s["a"].nunique(dropna=True) <= 1 or s["b"].nunique(dropna=True) <= 1:
        return float("nan")
    try:
        return float(s["a"].corr(s["b"], method=method))
    except Exception:
        return float("nan")


def make_deciles(score: pd.Series, n_bins: int = 10) -> pd.Series:
    ranked = pd.to_numeric(score, errors="coerce").rank(method="first", pct=True, ascending=True)
    deciles = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
    return deciles.astype("Int64") + 1  # 1=最低预测，10=最高预测


def build_decile_report(df_pred: pd.DataFrame, score_col: str, score_label: str) -> pd.DataFrame:
    d = df_pred.copy()
    d["decile"] = make_deciles(d[score_col], n_bins=10)
    agg_cols = [c for c in [
        "quality_score_short", "high_quality_flag_short",
        "ret_3", "ret_5", "mae_3", "mae_5", "rr_3", "rr_5"
    ] if c in d.columns]
    rows = []
    for decile, grp in d.groupby("decile", dropna=True):
        row: Dict[str, object] = {
            "decile": int(decile),
            "bucket_order_note": "1=lowest_pred,10=highest_pred",
            "count": int(len(grp)),
            f"{score_label}_mean": float(pd.to_numeric(grp[score_col], errors="coerce").mean()),
            f"{score_label}_min": float(pd.to_numeric(grp[score_col], errors="coerce").min()),
            f"{score_label}_max": float(pd.to_numeric(grp[score_col], errors="coerce").max()),
        }
        for col in agg_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(grp[col], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
    if not out.empty:
        for col in ["quality_score_short_mean", "ret_5_mean", "high_quality_flag_short_mean"]:
            if col in out.columns:
                out[f"{col}_spread_vs_bottom"] = out[col] - out[col].iloc[0]
                out[f"{col}_spread_vs_top"] = out[col] - out[col].iloc[-1]
        if "quality_score_short_mean" in out.columns:
            out["quality_score_monotonic_diff"] = out["quality_score_short_mean"].diff()
    return out


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    yt = pd.to_numeric(pd.Series(np.asarray(y_true)).reset_index(drop=True), errors="coerce")
    yp = pd.to_numeric(pd.Series(np.asarray(y_pred)).reset_index(drop=True), errors="coerce")
    valid = pd.DataFrame({"y_true": yt, "y_pred": yp}, index=range(len(yt))).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_corr": float("nan"), "spearman_corr": float("nan")}
    return {
        "mae": float(mean_absolute_error(valid["y_true"], valid["y_pred"])),
        "rmse": float(math.sqrt(mean_squared_error(valid["y_true"], valid["y_pred"]))),
        "pearson_corr": safe_corr(valid["y_true"].reset_index(drop=True), valid["y_pred"].reset_index(drop=True), method="pearson"),
        "spearman_corr": safe_corr(valid["y_true"].reset_index(drop=True), valid["y_pred"].reset_index(drop=True), method="spearman"),
    }


def compute_classification_metrics(y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    eps = 1e-12
    y_proba = np.clip(pd.to_numeric(pd.Series(y_proba), errors="coerce").to_numpy(dtype=float), eps, 1 - eps)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "logloss": float(log_loss(y_true, y_proba)),
    }


def build_feature_importance(model, feature_cols: Sequence[str]) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    return pd.DataFrame({
        "feature": list(feature_cols),
        "gain_importance": gain,
        "split_importance": split,
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)


def build_prediction_diagnostics(y_true: pd.Series, y_pred: Sequence[float], prefix: str) -> pd.DataFrame:
    yt = pd.to_numeric(pd.Series(np.asarray(y_true)).reset_index(drop=True), errors="coerce")
    yp = pd.to_numeric(pd.Series(np.asarray(y_pred)).reset_index(drop=True), errors="coerce")
    valid = pd.DataFrame({"y_true": yt, "y_pred": yp}, index=range(len(yt))).replace([np.inf, -np.inf], np.nan)
    rows = [
        {"item": f"{prefix}_rows", "value": int(len(valid))},
        {"item": f"{prefix}_valid_rows", "value": int(valid.dropna().shape[0])},
        {"item": f"{prefix}_y_true_std", "value": float(valid["y_true"].std(skipna=True))},
        {"item": f"{prefix}_y_pred_std", "value": float(valid["y_pred"].std(skipna=True))},
        {"item": f"{prefix}_y_true_mean", "value": float(valid["y_true"].mean(skipna=True))},
        {"item": f"{prefix}_y_pred_mean", "value": float(valid["y_pred"].mean(skipna=True))},
        {"item": f"{prefix}_y_true_q05", "value": float(valid["y_true"].quantile(0.05)) if valid["y_true"].notna().any() else np.nan},
        {"item": f"{prefix}_y_true_q95", "value": float(valid["y_true"].quantile(0.95)) if valid["y_true"].notna().any() else np.nan},
        {"item": f"{prefix}_y_pred_q05", "value": float(valid["y_pred"].quantile(0.05)) if valid["y_pred"].notna().any() else np.nan},
        {"item": f"{prefix}_y_pred_q95", "value": float(valid["y_pred"].quantile(0.95)) if valid["y_pred"].notna().any() else np.nan},
        {"item": f"{prefix}_y_true_nunique", "value": int(valid["y_true"].nunique(dropna=True))},
        {"item": f"{prefix}_y_pred_nunique", "value": int(valid["y_pred"].nunique(dropna=True))},
    ]
    return pd.DataFrame(rows)


def write_excel_sheets(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=name[:31], index=False)


def select_xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data = df[df[target_col].notna()].sort_values("trigger_dt").reset_index(drop=True)
    feature_cols_refined, _ = refine_feature_columns_for_source(data, feature_cols)
    X = data[list(feature_cols_refined)].copy()
    y = pd.to_numeric(data[target_col], errors="coerce")
    return X, y, data


def run_regression(df, feature_cols, split, out_dir, prefix, num_boost_round, early_stopping_rounds) -> Dict[str, object]:
    X, y, data = select_xy(df, feature_cols, "quality_score_short")
    train_idx, valid_idx, test_idx = split.train_idx, split.valid_idx, split.test_idx
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    model = fit_lightgbm_regression(X_train, y_train, X_valid, y_valid, REG_DEFAULT_PARAMS, num_boost_round, early_stopping_rounds)
    pred = model.predict(X_test, num_iteration=model.best_iteration)
    metrics = compute_regression_metrics(y_test, pred)
    pred_df = data.iloc[test_idx].copy()
    pred_df["pred_quality_score_short"] = pred
    deciles = build_decile_report(pred_df, "pred_quality_score_short", "pred_quality_score_short")
    fi = build_feature_importance(model, X.columns.tolist())
    diagnostics = build_prediction_diagnostics(y_test, pred, "reg_test")
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_excel_sheets(
        os.path.join(out_dir, f"{prefix}_deciles.xlsx"),
        {
            "deciles": deciles,
            "prediction_diagnostics": diagnostics,
            "test_predictions_preview": pred_df.head(500),
        },
    )
    fi.to_excel(os.path.join(out_dir, f"feature_importance_reg_{split.name}.xlsx"), index=False)
    return {"metrics": metrics, "deciles": deciles, "feature_importance": fi, "diagnostics": diagnostics}


def run_classification(df, feature_cols, split, out_dir, prefix, num_boost_round, early_stopping_rounds) -> Dict[str, object]:
    X, y, data = select_xy(df, feature_cols, "high_quality_flag_short")
    y = y.astype(int)
    train_idx, valid_idx, test_idx = split.train_idx, split.valid_idx, split.test_idx
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    model = fit_lightgbm_classification(X_train, y_train, X_valid, y_valid, CLF_DEFAULT_PARAMS, num_boost_round, early_stopping_rounds)
    proba = model.predict(X_test, num_iteration=model.best_iteration)
    metrics = compute_classification_metrics(y_test, proba)
    pred_df = data.iloc[test_idx].copy()
    pred_df["pred_high_quality_proba"] = proba
    deciles = build_decile_report(pred_df, "pred_high_quality_proba", "pred_high_quality_proba")
    fi = build_feature_importance(model, X.columns.tolist())
    diagnostics = build_prediction_diagnostics(y_test, proba, "clf_test")
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_excel_sheets(
        os.path.join(out_dir, f"{prefix}_deciles.xlsx"),
        {
            "deciles": deciles,
            "prediction_diagnostics": diagnostics,
            "test_predictions_preview": pred_df.head(500),
        },
    )
    fi.to_excel(os.path.join(out_dir, f"feature_importance_clf_{split.name}.xlsx"), index=False)
    return {"metrics": metrics, "deciles": deciles, "feature_importance": fi, "diagnostics": diagnostics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 3–5 天买点质量 GBDT 基线模型（精简特征版）")
    parser.add_argument("--model-table", required=True, help="训练表 parquet/xlsx/pkl 路径")
    parser.add_argument("--out-dir", default="gbdt_short_quality_output_v6", help="输出目录")
    parser.add_argument("--split-mode", choices=["single", "rolling"], default="single", help="时间切分模式")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="single 模式训练集占比")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="single 模式验证集占比")
    parser.add_argument("--n-folds", type=int, default=3, help="rolling 模式折数")
    parser.add_argument("--exclude-up-bars", action="store_true", help="轻量对照开关：排除 up_bars")
    parser.add_argument("--exclude-trigger-macd", action="store_true", help="轻量对照开关：排除 trigger_macd")
    parser.add_argument("--max-missing-ratio", type=float, default=0.60, help="特征最大缺失率")
    parser.add_argument("--num-boost-round", type=int, default=2000, help="最大 boosting 轮数")
    parser.add_argument("--early-stopping-rounds", type=int, default=150, help="early stopping 轮数")
    parser.add_argument("--tasks", default="reg,clf", help="任务，逗号分隔: reg,clf")
    parser.add_argument("--no-unify-clf-with-score", action="store_true", help="分类任务不与 score 样本口径统一")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    df = load_table(args.model_table)
    df = prepare_targets(df, unify_clf_with_score=not args.no_unify_clf_with_score)

    base_feature_cols, feature_stats = infer_feature_columns(df, max_missing_ratio=args.max_missing_ratio)
    summary = build_trainset_summary(df, feature_stats, base_feature_cols)
    yearly_dist, monthly_dist = build_time_distribution(df)
    feature_decision = build_feature_decision_table(df, exclude_up_bars=args.exclude_up_bars, exclude_trigger_macd=args.exclude_trigger_macd)
    excluded_preview = feature_decision[feature_decision["decision"] == "drop"].copy()

    write_excel_sheets(
        os.path.join(args.out_dir, "trainset_summary.xlsx"),
        {
            "summary": summary,
            "yearly_distribution": yearly_dist,
            "monthly_distribution": monthly_dist,
            "excluded_feature_preview": excluded_preview.head(300),
            "feature_decision": feature_decision,
        },
    )
    feature_stats.to_excel(os.path.join(args.out_dir, "feature_stats.xlsx"), index=False)

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    if args.split_mode == "single":
        reg_source = df[df["quality_score_short"].notna()].sort_values("trigger_dt").reset_index(drop=True)
        if "reg" in tasks and len(reg_source) > 10:
            split_reg = build_single_time_split(reg_source, args.train_ratio, args.valid_ratio)
            run_regression(reg_source, base_feature_cols, split_reg, args.out_dir, "lgb_reg", args.num_boost_round, args.early_stopping_rounds)

        if "clf" in tasks:
            clf_source = df if args.no_unify_clf_with_score else df[df["quality_score_short"].notna()].copy()
            clf_source = clf_source[clf_source["high_quality_flag_short"].notna()].sort_values("trigger_dt").reset_index(drop=True)
            if len(clf_source) > 10:
                split_clf = build_single_time_split(clf_source, args.train_ratio, args.valid_ratio)
                run_classification(clf_source, base_feature_cols, split_clf, args.out_dir, "lgb_clf", args.num_boost_round, args.early_stopping_rounds)
    else:
        rolling_rows: List[Dict[str, object]] = []
        if "reg" in tasks:
            reg_source = df[df["quality_score_short"].notna()].sort_values("trigger_dt").reset_index(drop=True)
            for split in build_rolling_splits(reg_source, args.n_folds):
                result = run_regression(reg_source, base_feature_cols, split, args.out_dir, f"lgb_reg_{split.name}", args.num_boost_round, args.early_stopping_rounds)
                rolling_rows.append({"task": "reg", "fold": split.name, **result["metrics"]})
        if "clf" in tasks:
            clf_source = df if args.no_unify_clf_with_score else df[df["quality_score_short"].notna()].copy()
            clf_source = clf_source[clf_source["high_quality_flag_short"].notna()].sort_values("trigger_dt").reset_index(drop=True)
            for split in build_rolling_splits(clf_source, args.n_folds):
                result = run_classification(clf_source, base_feature_cols, split, args.out_dir, f"lgb_clf_{split.name}", args.num_boost_round, args.early_stopping_rounds)
                rolling_rows.append({"task": "clf", "fold": split.name, **result["metrics"]})
        if rolling_rows:
            pd.DataFrame(rolling_rows).to_excel(os.path.join(args.out_dir, "rolling_summary.xlsx"), index=False)

    print("完成。输出目录:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
