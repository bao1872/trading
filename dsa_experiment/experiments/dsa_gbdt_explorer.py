#!/usr/bin/env python3
"""
DSA 选股 GBDT 双模型回测：分类（涨幅≥7%）+ 回归（涨跌幅）

Purpose: 训练 LightGBM 分类/回归模型，用 2023-2025 训练、2026 测试，输出完整回测指标
Inputs: dsa_selection 表, stock_k_data 表
Outputs: dsa_experiment/output/dsa_gbdt_*.csv

How to Run:
    python dsa_experiment/experiments/dsa_gbdt_explorer.py
    python dsa_experiment/experiments/dsa_gbdt_explorer.py --train-end 2024-12-31 --test-start 2025-01-01

Examples:
    python dsa_experiment/experiments/dsa_gbdt_explorer.py
    python dsa_experiment/experiments/dsa_gbdt_explorer.py --train-end 2024-12-31 --test-start 2025-01-01

Side Effects: 只读操作，输出 CSV 文件
"""

import sys
import os
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)

warnings.filterwarnings("ignore")

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

HORIZONS = [5, 10, 20, 40, 60]
PROFIT_THRESHOLD = 0.07  # 分类目标：mfe >= 7%
N_LAGS = 5  # 连续 5 bar 特征

SELECTED_FEATURES = [
    "regime_strength", "dsa_dir_bars",
    "offset_rate", "offset_mean", "offset_std", "offset_percentile",
    "vwap_ret_5", "vwap_ret_10", "vwap_ret_20",
    "cross_up_count",
    "change_pct", "vol_zscore", "avg_amount_20d",
    "rope_dir1_pct", "rope_dir0_pct", "rope_dir_neg1_pct",
    "touch_rope", "touch_vwap",
]

CLS_PARAMS = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 16, "max_depth": 5, "min_child_samples": 50,
    "learning_rate": 0.03, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "max_bin": 63, "seed": 42, "verbosity": -1,
}

REG_PARAMS = {
    "objective": "regression", "metric": "mae",
    "num_leaves": 16, "max_depth": 5, "min_child_samples": 50,
    "learning_rate": 0.03, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "max_bin": 63, "seed": 42, "verbosity": -1,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50


def load_data(start_date: str = None) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        sql = "SELECT * FROM dsa_selection"
        if start_date:
            sql += f" WHERE selection_date >= '{start_date}'"
        sql += " ORDER BY selection_date, ts_code"
        df = pd.read_sql(sql, conn)
    print(f"  dsa_selection 记录数: {len(df)}")
    return df


def compute_future_returns(sel_df: pd.DataFrame) -> pd.DataFrame:
    """向量化按股票计算未来收益"""
    engine = create_engine(DATABASE_URL)
    sel_df = sel_df.copy()
    sel_df["selection_date"] = pd.to_datetime(sel_df["selection_date"]).dt.normalize()

    label_cols = []
    for n in HORIZONS:
        label_cols.extend([f"ret_{n}", f"mae_{n}", f"mfe_{n}"])
    for col in label_cols:
        sel_df[col] = np.nan

    ts_codes = sel_df["ts_code"].unique().tolist()
    print(f"  需处理 {len(ts_codes)} 只股票 ...")

    processed = 0
    for ts_code in ts_codes:
        sql = f"SELECT bar_time, open, high, low, close FROM stock_k_data WHERE ts_code = '{ts_code}' AND freq = 'd' ORDER BY bar_time"
        with engine.connect() as conn:
            kline = pd.read_sql(sql, conn)
        if len(kline) < 20:
            continue

        kline["bar_time"] = pd.to_datetime(kline["bar_time"]).dt.normalize()
        kline = kline.set_index("bar_time")

        mask = sel_df["ts_code"] == ts_code
        stock_rows = sel_df.loc[mask]
        if stock_rows.empty:
            continue

        close_s = kline["close"]
        high_s = kline["high"]
        low_s = kline["low"]

        for n in HORIZONS:
            ret_n = close_s.shift(-n) / close_s - 1
            mae_n = low_s.shift(-1).rolling(n, min_periods=1).min().shift(-(n - 1)) / close_s - 1
            mfe_n = high_s.shift(-1).rolling(n, min_periods=1).max().shift(-(n - 1)) / close_s - 1

            dates = stock_rows["selection_date"]
            valid_dates = dates[dates.isin(ret_n.index)]
            if valid_dates.empty:
                continue
            sel_df.loc[valid_dates.index, f"ret_{n}"] = ret_n.loc[valid_dates].values
            sel_df.loc[valid_dates.index, f"mae_{n}"] = mae_n.loc[valid_dates].values
            sel_df.loc[valid_dates.index, f"mfe_{n}"] = mfe_n.loc[valid_dates].values

        processed += 1
        if processed % 500 == 0:
            print(f"    已处理 {processed}/{len(ts_codes)} 只股票")

    print(f"  已处理 {processed}/{len(ts_codes)} 只股票")
    return sel_df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造连续 5 bar 特征：对每只股票，将 T-4..T 的特征拼接"""
    df = df.sort_values(["ts_code", "selection_date"]).copy()

    lag_dfs = []
    for lag in range(N_LAGS):
        lag_cols = {}
        for feat in SELECTED_FEATURES:
            col_name = f"{feat}_t{lag}"
            if feat in df.columns:
                shifted = df.groupby("ts_code")[feat].shift(lag)
                if shifted.dtype == bool or shifted.dtype == object:
                    shifted = shifted.astype(float)
                lag_cols[col_name] = shifted
            else:
                lag_cols[col_name] = np.nan
        lag_df = pd.DataFrame(lag_cols, index=df.index)
        lag_dfs.append(lag_df)

    lag_result = pd.concat(lag_dfs, axis=1)
    df = pd.concat([df, lag_result], axis=1)

    lag_feature_cols = [f"{feat}_t{lag}" for feat in SELECTED_FEATURES for lag in range(N_LAGS)]
    print(f"  连续 {N_LAGS} bar 特征数: {len(lag_feature_cols)}")
    return df, lag_feature_cols


# ---------------------------------------------------------------------------
# 分类模型
# ---------------------------------------------------------------------------
def train_cls_model(train_df, test_df, feature_cols, label_col):
    """训练分类模型，返回指标 + 测试集预测"""
    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    valid_train = X_train.notna().any(axis=1) & y_train.notna()
    valid_test = X_test.notna().any(axis=1) & y_test.notna()
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    if len(y_train) < 100 or len(y_test) < 30 or y_train.nunique() < 2:
        return None

    pos_rate = y_train.mean()
    if pos_rate < 0.01 or pos_rate > 0.99:
        return None

    model = lgb.LGBMClassifier(
        n_estimators=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **CLS_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    importance = dict(zip(feature_cols, model.feature_importances_))

    # 测试集带预测的结果
    test_result = test_df[valid_test].copy()
    test_result["pred_prob"] = y_pred_prob
    test_result["pred_label"] = y_pred
    test_result["actual_label"] = y_test.values

    return {
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "pos_rate": round(float(pos_rate), 4),
        "train_n": len(y_train),
        "test_n": len(y_test),
        "importance": importance,
        "best_iteration": model.best_iteration_,
        "test_result": test_result,
    }


# ---------------------------------------------------------------------------
# 回归模型
# ---------------------------------------------------------------------------
def train_reg_model(train_df, test_df, feature_cols, label_col):
    """训练回归模型，返回指标 + 测试集预测"""
    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    valid_train = X_train.notna().any(axis=1) & y_train.notna()
    valid_test = X_test.notna().any(axis=1) & y_test.notna()
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    if len(y_train) < 100 or len(y_test) < 30:
        return None

    model = lgb.LGBMRegressor(
        n_estimators=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **REG_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = model.predict(X_test)

    # 日频截面 IC
    test_info = test_df[valid_test].copy()
    test_info["pred"] = y_pred
    test_info["label"] = y_test.values
    daily_ics = []
    for dt, day_df in test_info.groupby("selection_date"):
        if len(day_df) < 5:
            continue
        ic_val, _ = stats.spearmanr(day_df["pred"], day_df["label"])
        if np.isfinite(ic_val):
            daily_ics.append(ic_val)

    mean_ic = np.mean(daily_ics) if daily_ics else np.nan
    std_ic = np.std(daily_ics) if daily_ics else np.nan
    icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan

    mae_val = mean_absolute_error(y_test, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_val = r2_score(y_test, y_pred)

    # 方向准确率
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_test.values))

    importance = dict(zip(feature_cols, model.feature_importances_))

    # 测试集带预测的结果
    test_result = test_df[valid_test].copy()
    test_result["pred"] = y_pred
    test_result["actual"] = y_test.values

    return {
        "ic": round(mean_ic, 4) if np.isfinite(mean_ic) else np.nan,
        "icir": round(icir, 4) if np.isfinite(icir) else np.nan,
        "mae": round(mae_val, 4),
        "rmse": round(rmse_val, 4),
        "r2": round(r2_val, 4),
        "direction_acc": round(direction_acc, 4),
        "train_n": len(y_train),
        "test_n": len(y_test),
        "importance": importance,
        "best_iteration": model.best_iteration_,
        "test_result": test_result,
    }


# ---------------------------------------------------------------------------
# 回测指标计算
# ---------------------------------------------------------------------------
def compute_quintile_backtest(test_result: pd.DataFrame, pred_col: str,
                               return_col: str, mae_col: str = None,
                               mfe_col: str = None, n_groups: int = 5,
                               horizon: int = 20) -> dict:
    """
    按预测分组计算回测指标：
    - 每日截面分组，计算各组实际收益
    - Top 组组合指标：年化收益、夏普、最大回撤、胜率
    - 多空对冲指标
    """
    df = test_result.dropna(subset=[pred_col, return_col]).copy()
    if len(df) < 50:
        return None

    # 每日截面分组
    df["quintile"] = np.nan
    for dt, idx in df.groupby("selection_date").groups.items():
        day_data = df.loc[idx, pred_col]
        if len(day_data) < n_groups:
            continue
        try:
            df.loc[idx, "quintile"] = pd.qcut(day_data, n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue

    df = df.dropna(subset=["quintile"])
    df["quintile"] = df["quintile"].astype(int)

    # ---- 各分组平均收益 ----
    quintile_stats = {}
    for q in range(n_groups):
        q_df = df[df["quintile"] == q]
        stats_q = {
            "count": len(q_df),
            "avg_return": round(q_df[return_col].mean(), 4),
            "median_return": round(q_df[return_col].median(), 4),
            "hit_rate": round((q_df[return_col] > 0).mean(), 4),
        }
        if mae_col and mae_col in df.columns:
            stats_q["avg_mae"] = round(q_df[mae_col].mean(), 4)
        if mfe_col and mfe_col in df.columns:
            stats_q["avg_mfe"] = round(q_df[mfe_col].mean(), 4)
        quintile_stats[q] = stats_q

    # ---- 每日 Top 组收益序列 ----
    top_q = n_groups - 1
    bottom_q = 0
    daily_top_ret = df[df["quintile"] == top_q].groupby("selection_date")[return_col].mean()
    daily_bottom_ret = df[df["quintile"] == bottom_q].groupby("selection_date")[return_col].mean()
    daily_ls_ret = daily_top_ret - daily_bottom_ret

    # ---- Top 组组合指标 ----
    top_metrics = _compute_portfolio_metrics(daily_top_ret, "top", horizon)
    bottom_metrics = _compute_portfolio_metrics(daily_bottom_ret, "bottom", horizon)
    ls_metrics = _compute_portfolio_metrics(daily_ls_ret, "long_short", horizon)

    # ---- 单调性检验 ----
    q_avg_returns = [quintile_stats[q]["avg_return"] for q in range(n_groups)]
    monotonic = all(q_avg_returns[i] <= q_avg_returns[i + 1] for i in range(len(q_avg_returns) - 1))
    # Spearman 秩相关
    spearman_r, spearman_p = stats.spearmanr(list(range(n_groups)), q_avg_returns)

    return {
        "quintile_stats": quintile_stats,
        "top_metrics": top_metrics,
        "bottom_metrics": bottom_metrics,
        "ls_metrics": ls_metrics,
        "monotonic": monotonic,
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "daily_top_ret": daily_top_ret,
        "daily_ls_ret": daily_ls_ret,
    }


def _compute_portfolio_metrics(daily_ret: pd.Series, name: str, horizon: int = 20) -> dict:
    """计算组合级别指标（daily_ret 为 N 日收益序列）"""
    if len(daily_ret) < 5:
        return {}

    # N 日收益 → 等效日收益
    eq_daily_ret = (1 + daily_ret) ** (1 / horizon) - 1

    cum_ret = (1 + eq_daily_ret).cumprod()
    total_ret = cum_ret.iloc[-1] - 1

    n_days = len(eq_daily_ret)
    ann_ret = (1 + total_ret) ** (242 / max(n_days, 1)) - 1

    ann_vol = eq_daily_ret.std() * np.sqrt(242)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    max_dd = drawdown.min()

    # 胜率（等效日收益 > 0）
    win_rate = (eq_daily_ret > 0).mean()

    # 盈亏比
    avg_win = eq_daily_ret[eq_daily_ret > 0].mean() if (eq_daily_ret > 0).any() else 0
    avg_loss = abs(eq_daily_ret[eq_daily_ret < 0].mean()) if (eq_daily_ret < 0).any() else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Sortino（只考虑下行波动）
    downside = eq_daily_ret[eq_daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(242) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    # 平均 N 日收益（原始口径）
    avg_n_day_ret = daily_ret.mean()

    return {
        "total_return": round(total_ret, 4),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_loss_ratio": round(profit_loss_ratio, 4),
        "avg_n_day_ret": round(avg_n_day_ret, 4),
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# 特征重要性分析
# ---------------------------------------------------------------------------
def analyze_feature_importance(all_importance: dict) -> pd.DataFrame:
    rows = []
    for feat in SELECTED_FEATURES:
        row = {"feature": feat}
        total_gain = 0
        for lag in range(N_LAGS):
            col = f"{feat}_t{lag}"
            gain = sum(imp.get(col, 0) for imp in all_importance.values())
            row[f"t{lag}_gain"] = gain
            total_gain += gain
        row["total_gain"] = total_gain
        if total_gain > 0:
            for lag in range(N_LAGS):
                row[f"t{lag}_pct"] = round(row[f"t{lag}_gain"] / total_gain * 100, 1)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("total_gain", ascending=False)
    return df


def analyze_timestep_contribution(all_importance: dict) -> dict:
    timestep_gains = {f"t{lag}": 0 for lag in range(N_LAGS)}
    for feat in SELECTED_FEATURES:
        for lag in range(N_LAGS):
            col = f"{feat}_t{lag}"
            gain = sum(imp.get(col, 0) for imp in all_importance.values())
            timestep_gains[f"t{lag}"] += gain
    total = sum(timestep_gains.values())
    if total == 0:
        return timestep_gains
    return {k: round(v / total * 100, 1) for k, v in timestep_gains.items()}


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DSA 选股 GBDT 双模型回测")
    parser.add_argument("--train-end", type=str, default="2025-12-31",
                        help="训练集截止日期 (默认 2025-12-31)")
    parser.add_argument("--test-start", type=str, default="2026-01-01",
                        help="测试集起始日期 (默认 2026-01-01)")
    args = parser.parse_args()

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("DSA 选股 GBDT 双模型回测")
    print(f"  训练集: ~ {args.train_end}  |  测试集: {args.test_start} ~")
    print("=" * 80)

    # [1/5] 加载数据
    print("\n[1/5] 加载数据 ...")
    sel_df = load_data()
    if sel_df.empty:
        print("  无数据，退出")
        return

    sel_df["selection_date"] = pd.to_datetime(sel_df["selection_date"]).dt.normalize()

    # [2/5] 计算未来收益
    print("\n[2/5] 计算未来收益 ...")
    df = compute_future_returns(sel_df)

    # [3/5] 构造连续 5 bar 特征
    print("\n[3/5] 构造连续 5 bar 特征 ...")
    df, lag_feature_cols = build_lag_features(df)

    # 年份分割
    train_df = df[df["selection_date"] <= train_end]
    test_df = df[df["selection_date"] >= test_start]
    print(f"\n  训练集: {len(train_df)} 条 ({train_df['selection_date'].min().date()} ~ {train_df['selection_date'].max().date()})")
    print(f"  测试集: {len(test_df)} 条 ({test_df['selection_date'].min().date()} ~ {test_df['selection_date'].max().date()})")

    if len(train_df) < 200 or len(test_df) < 50:
        print("  样本不足，退出")
        return

    # ===== 分类模型 =====
    print("\n" + "=" * 80)
    print("[4/6] 分类模型（mfe >= 7%）")
    print("=" * 80)

    cls_results = []
    all_cls_importance = {}
    cls_backtest_results = {}

    for n in HORIZONS:
        label_col = f"mfe_{n}"
        cls_label = f"cls_mfe_{n}"
        ret_col = f"ret_{n}"
        mae_col = f"mae_{n}"

        train_df_c = train_df.copy()
        test_df_c = test_df.copy()
        train_df_c[cls_label] = (train_df_c[label_col] >= PROFIT_THRESHOLD).astype(int)
        test_df_c[cls_label] = (test_df_c[label_col] >= PROFIT_THRESHOLD).astype(int)

        result = train_cls_model(train_df_c, test_df_c, lag_feature_cols, cls_label)
        if result is None:
            print(f"  Horizon={n}: 样本不足，跳过")
            continue

        result["horizon"] = n
        cls_results.append(result)
        all_cls_importance[f"cls_{n}"] = result["importance"]

        # 回测指标
        test_res = result.pop("test_result")
        bt = compute_quintile_backtest(
            test_res, "pred_prob", ret_col, mae_col, label_col, horizon=n
        )
        cls_backtest_results[n] = bt

        # 打印模型指标
        print(f"\n  --- Horizon={n}天 ---")
        print(f"  AUC={result['auc']:.4f}  Acc={result['accuracy']:.4f}  "
              f"Prec={result['precision']:.4f}  Rec={result['recall']:.4f}  F1={result['f1']:.4f}")
        print(f"  正样本率={result['pos_rate']:.3f}  训练={result['train_n']}  测试={result['test_n']}  BestIter={result['best_iteration']}")
        cm = result["confusion_matrix"]
        if cm and len(cm) >= 2:
            print(f"  混淆矩阵: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

        if bt:
            print(f"  分组收益 (avg_ret | hit_rate):")
            for q in range(5):
                qs = bt["quintile_stats"].get(q, {})
                print(f"    Q{q+1}: ret={qs.get('avg_return', 'N/A'):.4f}  "
                      f"hit={qs.get('hit_rate', 'N/A'):.4f}  "
                      f"mae={qs.get('avg_mae', 'N/A'):.4f}  "
                      f"mfe={qs.get('avg_mfe', 'N/A'):.4f}")
            tm = bt["top_metrics"]
            lm = bt["ls_metrics"]
            print(f"  Top组: 年化={tm.get('ann_return', 0):.4f}  夏普={tm.get('sharpe', 0):.4f}  "
                  f"最大回撤={tm.get('max_drawdown', 0):.4f}  胜率={tm.get('win_rate', 0):.4f}  "
                  f"盈亏比={tm.get('profit_loss_ratio', 0):.4f}")
            print(f"  多空: 年化={lm.get('ann_return', 0):.4f}  夏普={lm.get('sharpe', 0):.4f}  "
                  f"最大回撤={lm.get('max_drawdown', 0):.4f}")
            print(f"  单调性={bt['monotonic']}  SpearmanR={bt['spearman_r']:.4f}  p={bt['spearman_p']:.4f}")

    # ===== 回归模型 =====
    print("\n" + "=" * 80)
    print("[5/6] 回归模型（ret）")
    print("=" * 80)

    reg_results = []
    all_reg_importance = {}
    reg_backtest_results = {}

    for n in HORIZONS:
        label_col = f"ret_{n}"
        mae_col = f"mae_{n}"
        mfe_col = f"mfe_{n}"

        result = train_reg_model(train_df, test_df, lag_feature_cols, label_col)
        if result is None:
            print(f"  Horizon={n}: 样本不足，跳过")
            continue

        result["horizon"] = n
        reg_results.append(result)
        all_reg_importance[f"reg_{n}"] = result["importance"]

        # 回测指标
        test_res = result.pop("test_result")
        bt = compute_quintile_backtest(
            test_res, "pred", label_col, mae_col, mfe_col, horizon=n
        )
        reg_backtest_results[n] = bt

        # 打印模型指标
        print(f"\n  --- Horizon={n}天 ---")
        print(f"  IC={result['ic']:.4f}  ICIR={result['icir']:.4f}  "
              f"MAE={result['mae']:.4f}  RMSE={result['rmse']:.4f}  "
              f"R²={result['r2']:.4f}  方向准确率={result['direction_acc']:.4f}")
        print(f"  训练={result['train_n']}  测试={result['test_n']}  BestIter={result['best_iteration']}")

        if bt:
            print(f"  分组收益 (avg_ret | hit_rate):")
            for q in range(5):
                qs = bt["quintile_stats"].get(q, {})
                print(f"    Q{q+1}: ret={qs.get('avg_return', 'N/A'):.4f}  "
                      f"hit={qs.get('hit_rate', 'N/A'):.4f}  "
                      f"mae={qs.get('avg_mae', 'N/A'):.4f}  "
                      f"mfe={qs.get('avg_mfe', 'N/A'):.4f}")
            tm = bt["top_metrics"]
            lm = bt["ls_metrics"]
            print(f"  Top组: 年化={tm.get('ann_return', 0):.4f}  夏普={tm.get('sharpe', 0):.4f}  "
                  f"最大回撤={tm.get('max_drawdown', 0):.4f}  胜率={tm.get('win_rate', 0):.4f}  "
                  f"盈亏比={tm.get('profit_loss_ratio', 0):.4f}")
            print(f"  多空: 年化={lm.get('ann_return', 0):.4f}  夏普={lm.get('sharpe', 0):.4f}  "
                  f"最大回撤={lm.get('max_drawdown', 0):.4f}")
            print(f"  单调性={bt['monotonic']}  SpearmanR={bt['spearman_r']:.4f}  p={bt['spearman_p']:.4f}")

    # ===== 特征重要性 =====
    print("\n" + "=" * 80)
    print("[6/6] 特征重要性分析")
    print("=" * 80)

    all_importance = {**all_cls_importance, **all_reg_importance}

    if all_importance:
        feat_imp_df = analyze_feature_importance(all_importance)
        print("\n  Top 15 特征:")
        for _, row in feat_imp_df.head(15).iterrows():
            print(f"    {row['feature']:<25} total={row['total_gain']:>6.0f}  "
                  f"t0={row.get('t0_pct', 0):>5.1f}%  t1={row.get('t1_pct', 0):>5.1f}%  "
                  f"t2={row.get('t2_pct', 0):>5.1f}%  t3={row.get('t3_pct', 0):>5.1f}%  "
                  f"t4={row.get('t4_pct', 0):>5.1f}%")

        ts_contrib = analyze_timestep_contribution(all_importance)
        print(f"\n  时间步贡献: {ts_contrib}")

        feat_imp_path = os.path.join(OUTPUT_DIR, "dsa_gbdt_feature_importance.csv")
        feat_imp_df.to_csv(feat_imp_path, index=False)
        print(f"  特征重要性已保存: {feat_imp_path}")

    # ===== 保存结果 =====
    # 分类模型汇总表
    if cls_results:
        cls_rows = []
        for r in cls_results:
            row = {k: v for k, v in r.items() if k not in ["importance", "confusion_matrix"]}
            n = r["horizon"]
            bt = cls_backtest_results.get(n, {})
            if bt:
                tm = bt.get("top_metrics", {})
                lm = bt.get("ls_metrics", {})
                row["top_ann_ret"] = tm.get("ann_return")
                row["top_sharpe"] = tm.get("sharpe")
                row["top_max_dd"] = tm.get("max_drawdown")
                row["top_win_rate"] = tm.get("win_rate")
                row["top_pl_ratio"] = tm.get("profit_loss_ratio")
                row["ls_ann_ret"] = lm.get("ann_return")
                row["ls_sharpe"] = lm.get("sharpe")
                row["ls_max_dd"] = lm.get("max_drawdown")
                row["monotonic"] = bt.get("monotonic")
                row["spearman_r"] = bt.get("spearman_r")
            cls_rows.append(row)
        cls_df = pd.DataFrame(cls_rows)
        cls_path = os.path.join(OUTPUT_DIR, "dsa_gbdt_cls_results.csv")
        cls_df.to_csv(cls_path, index=False)
        print(f"\n  分类结果已保存: {cls_path}")

    # 回归模型汇总表
    if reg_results:
        reg_rows = []
        for r in reg_results:
            row = {k: v for k, v in r.items() if k not in ["importance"]}
            n = r["horizon"]
            bt = reg_backtest_results.get(n, {})
            if bt:
                tm = bt.get("top_metrics", {})
                lm = bt.get("ls_metrics", {})
                row["top_ann_ret"] = tm.get("ann_return")
                row["top_sharpe"] = tm.get("sharpe")
                row["top_max_dd"] = tm.get("max_drawdown")
                row["top_win_rate"] = tm.get("win_rate")
                row["top_pl_ratio"] = tm.get("profit_loss_ratio")
                row["ls_ann_ret"] = lm.get("ann_return")
                row["ls_sharpe"] = lm.get("sharpe")
                row["ls_max_dd"] = lm.get("max_drawdown")
                row["monotonic"] = bt.get("monotonic")
                row["spearman_r"] = bt.get("spearman_r")
            reg_rows.append(row)
        reg_df = pd.DataFrame(reg_rows)
        reg_path = os.path.join(OUTPUT_DIR, "dsa_gbdt_reg_results.csv")
        reg_df.to_csv(reg_path, index=False)
        print(f"  回归结果已保存: {reg_path}")

    # 分组明细表
    for model_type, bt_results in [("cls", cls_backtest_results), ("reg", reg_backtest_results)]:
        q_rows = []
        for n, bt in bt_results.items():
            if not bt:
                continue
            for q in range(5):
                qs = bt["quintile_stats"].get(q, {})
                q_rows.append({
                    "model": model_type, "horizon": n, "quintile": q + 1,
                    **qs,
                })
        if q_rows:
            q_df = pd.DataFrame(q_rows)
            q_path = os.path.join(OUTPUT_DIR, f"dsa_gbdt_{model_type}_quintile.csv")
            q_df.to_csv(q_path, index=False)
            print(f"  {model_type} 分组明细已保存: {q_path}")

    print("\n" + "=" * 80)
    print("回测完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
