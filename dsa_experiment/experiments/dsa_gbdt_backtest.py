#!/usr/bin/env python3
"""
DSA 选股 GBDT Walk-Forward 回测

Purpose: 分类模型（MFE≥7%），Walk-forward 验证，真实组合净值回测
Inputs: dsa_selection 表, stock_k_data 表
Outputs: dsa_experiment/output/dsa_gbdt_backtest_*.csv, dsa_experiment/output/dsa_gbdt/predict_*.csv

How to Run:
    python dsa_experiment/experiments/dsa_gbdt_backtest.py
    python dsa_experiment/experiments/dsa_gbdt_backtest.py --top-n 30
    python dsa_experiment/experiments/dsa_gbdt_backtest.py --predict-date 2026-06-02

Examples:
    python dsa_experiment/experiments/dsa_gbdt_backtest.py
    python dsa_experiment/experiments/dsa_gbdt_backtest.py --top-n 30
    python dsa_experiment/experiments/dsa_gbdt_backtest.py --predict-date 2026-06-02

Side Effects: 只读操作，输出 CSV 文件和模型文件
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
)

warnings.filterwarnings("ignore")

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# ===== 可配置参数 =====
HORIZONS = [5, 10]
PROFIT_THRESHOLD = 0.07  # MFE ≥ 7% 为正样本
TOP_N = 20               # 每期持仓数
BUY_COST = 0.001         # 买入成本 0.1%（佣金+滑点）
SELL_COST = 0.002        # 卖出成本 0.2%（佣金+滑点+印花税）
MIN_AMOUNT = 10_000_000  # 最低日成交额 1000万
N_LAGS = 5

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

NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 30


# ===========================================================================
# 数据加载
# ===========================================================================
def load_data() -> pd.DataFrame:
    """加载 dsa_selection 数据"""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT * FROM dsa_selection ORDER BY selection_date, ts_code", conn
        )
    print(f"  dsa_selection 记录数: {len(df)}")
    # 板块分布
    df["board"] = df["ts_code"].apply(_get_board)
    board_dist = df.groupby("board")["ts_code"].nunique().to_dict()
    print(f"  板块分布: {board_dist}")
    return df


def preload_kline(ts_codes: list, start_date: str = "2022-11-01") -> dict:
    """预加载 kline 数据，返回 {ts_code: DataFrame}

    只加载 start_date 之后的数据，减少内存占用。
    """
    engine = create_engine(DATABASE_URL)
    kline_dict = {}
    batch_size = 200
    total_records = 0
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i + batch_size]
        placeholders = ",".join([f"'{c}'" for c in batch])
        sql = (
            f"SELECT ts_code, bar_time, open, high, low, close, volume "
            f"FROM stock_k_data WHERE freq = 'd' AND ts_code IN ({placeholders}) "
            f"AND bar_time >= '{start_date}' "
            f"ORDER BY ts_code, bar_time"
        )
        with engine.connect() as conn:
            kline = pd.read_sql(sql, conn)
        if kline.empty:
            continue
        kline["bar_time"] = pd.to_datetime(kline["bar_time"]).dt.normalize()
        kline["amount"] = kline["close"] * kline["volume"]
        for ts_code, group in kline.groupby("ts_code"):
            g = group.set_index("bar_time").sort_index()
            kline_dict[ts_code] = g
            total_records += len(g)
        # 释放批次内存
        del kline

    print(f"  预加载 kline: {len(kline_dict)} 只股票, {total_records} 条记录")
    return kline_dict


# ===========================================================================
# 未来收益计算（MFE/MAE/RET）
# ===========================================================================
def compute_future_returns(kline_dict: dict, sel_df: pd.DataFrame,
                           horizons: list) -> pd.DataFrame:
    """
    计算未来收益：ret_N, mae_N, mfe_N
    向量化按股票计算，避免 OOM
    """
    sel_df = sel_df.copy()
    sel_df["selection_date"] = pd.to_datetime(sel_df["selection_date"]).dt.normalize()

    label_cols = []
    for n in horizons:
        label_cols.extend([f"ret_{n}", f"mae_{n}", f"mfe_{n}"])
    for col in label_cols:
        sel_df[col] = np.nan

    ts_codes = sel_df["ts_code"].unique().tolist()
    print(f"  需处理 {len(ts_codes)} 只股票 ...")

    processed = 0
    for ts_code in ts_codes:
        if ts_code not in kline_dict:
            continue
        kline = kline_dict[ts_code]
        if len(kline) < 20:
            continue

        close_s = kline["close"]
        high_s = kline["high"]
        low_s = kline["low"]

        mask = sel_df["ts_code"] == ts_code
        stock_rows = sel_df.loc[mask]
        if stock_rows.empty:
            continue

        for n in horizons:
            ret_n = close_s.shift(-n) / close_s - 1
            mae_n = low_s.shift(-1).rolling(n, min_periods=1).min().shift(-(n - 1)) / close_s - 1
            mfe_n = high_s.shift(-1).rolling(n, min_periods=1).max().shift(-(n - 1)) / close_s - 1

            dates = stock_rows["selection_date"]
            valid_dates = dates[dates.isin(close_s.index)]
            if valid_dates.empty:
                continue
            sel_df.loc[valid_dates.index, f"ret_{n}"] = ret_n.loc[valid_dates].values
            sel_df.loc[valid_dates.index, f"mae_{n}"] = mae_n.loc[valid_dates].values
            sel_df.loc[valid_dates.index, f"mfe_{n}"] = mfe_n.loc[valid_dates].values

        processed += 1
        if processed % 200 == 0:
            print(f"    已处理 {processed}/{len(ts_codes)} 只股票")

    print(f"  已处理 {processed}/{len(ts_codes)} 只股票")

    # 标签统计
    for n in horizons:
        mfe_col = f"mfe_{n}"
        valid = sel_df[mfe_col].notna()
        pos = (sel_df.loc[valid, mfe_col] >= PROFIT_THRESHOLD).sum()
        total = valid.sum()
        print(f"  H={n}: 有效样本={total}, MFE≥7%={pos} ({pos/total:.1%})" if total > 0 else f"  H={n}: 无有效样本")

    return sel_df


# ===========================================================================
# 特征工程
# ===========================================================================
def build_lag_features(df: pd.DataFrame, n_lags: int = N_LAGS) -> tuple:
    """构造连续 n_lags bar 特征（不修改原始 df，只返回 lag 列和列名）"""
    lag_dfs = []
    for lag in range(n_lags):
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
    lag_feature_cols = [f"{feat}_t{lag}" for feat in SELECTED_FEATURES for lag in range(n_lags)]
    print(f"  连续 {n_lags} bar 特征数: {len(lag_feature_cols)}")
    return lag_result, lag_feature_cols


# ===========================================================================
# 辅助函数
# ===========================================================================
def _get_board(ts_code: str) -> str:
    """获取板块"""
    if ts_code.startswith("688"):
        return "科创板"
    elif ts_code.startswith("30"):
        return "创业板"
    elif ts_code.startswith("00"):
        return "深主板"
    elif ts_code.startswith("60"):
        return "沪主板"
    else:
        return "其他"


def _get_limit_pct(ts_code: str) -> float:
    """获取涨跌停幅度"""
    if ts_code.startswith("30") or ts_code.startswith("688"):
        return 0.20
    return 0.10

def _get_trailing_cutoff(all_dates: list, test_start: pd.Timestamp,
                         n_days: int) -> pd.Timestamp:
    """获取训练集截止日期，确保标签不穿越到测试期

    标签 mfe_N 使用未来 N 天价格，训练集末尾 N 天的标签会穿越到测试期。
    将训练集截止日期提前 n_days 个交易日，消除标签穿越。
    """
    # 找到 test_start 在日期序列中的位置
    test_idx = -1
    for i, d in enumerate(all_dates):
        if d >= test_start:
            test_idx = i
            break
    if test_idx < 0:
        return all_dates[-1]
    cutoff_idx = max(0, test_idx - n_days)
    return all_dates[cutoff_idx]


# ===========================================================================
# Walk-Forward 训练
# ===========================================================================
def _train_cls_fold(train_df, test_df, feature_cols, label_col,
                    sample_weights=None, return_model=False):
    """训练单个折的分类模型

    从训练集尾部切 20% 做验证集用于 early stopping，
    测试集仅用于最终评估，不参与模型选择。

    Args:
        sample_weights: 训练样本权重，长度与 train_df 一致。
            切分验证集时同步切分权重。
        return_model: 是否返回训练好的模型对象（用于预测场景）。
    """
    X_all = train_df[feature_cols]
    y_all = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    valid_train = X_all.notna().any(axis=1) & y_all.notna()
    valid_test = X_test.notna().any(axis=1) & y_test.notna()
    X_all, y_all = X_all[valid_train], y_all[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    # 同步过滤 sample_weights
    if sample_weights is not None:
        sample_weights = sample_weights[valid_train.values]

    if len(y_all) < 100 or len(y_test) < 30 or y_all.nunique() < 2:
        return None, None

    pos_rate = y_all.mean()
    if pos_rate < 0.01 or pos_rate > 0.99:
        return None, None

    # 从训练集尾部切 20% 做验证集（按时间顺序，尾部更接近测试期）
    val_ratio = 0.2
    val_size = max(int(len(y_all) * val_ratio), 50)
    X_val, y_val = X_all.iloc[-val_size:], y_all.iloc[-val_size:]
    X_train, y_train = X_all.iloc[:-val_size], y_all.iloc[:-val_size]

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return None, None

    model = lgb.LGBMClassifier(
        n_estimators=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **CLS_PARAMS,
    )

    # 带权重训练
    if sample_weights is not None:
        w_train = sample_weights[:-val_size]
        w_val = sample_weights[-val_size:]
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], eval_sample_weight=[w_val])
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "auc": round(auc, 4), "accuracy": round(acc, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "pos_rate": round(float(pos_rate), 4),
        "train_n": len(y_train), "val_n": len(y_val),
        "test_n": len(y_test), "best_iter": model.best_iteration_,
    }

    oos_cols = ["selection_date", "ts_code", "stock_name", "board"]
    # 保留 ret/mae/mfe 列（portfolio_backtest 需要）
    horizon_num = label_col.split("_")[1] if label_col.startswith("mfe_") else None
    if horizon_num:
        for prefix in ["ret", "mae", "mfe"]:
            col = f"{prefix}_{horizon_num}"
            if col in test_df.columns:
                oos_cols.append(col)
    oos = test_df.loc[valid_test, oos_cols].copy()
    oos["pred_prob"] = y_pred_prob

    if return_model:
        return metrics, oos, model
    return metrics, oos

def walk_forward_rolling(df, lag_cols, horizons, window_quarters=4,
                         use_weight=False, window_name="rolling_4q"):
    """滚动窗口 Walk-Forward

    Args:
        window_quarters: 训练窗口季度数
            - None 或 0: expanding（全历史）
            - 2/4/6/8: 滚动窗口
        use_weight: 是否使用时间加权（近期权重更高）
        window_name: 窗口名称，用于输出标识
    """
    df["selection_date"] = pd.to_datetime(df["selection_date"]).dt.normalize()

    all_dates = sorted(df["selection_date"].unique())
    data_start = all_dates[0]

    # 提取所有季度
    quarters = []
    for d in all_dates:
        q_key = f"{d.year}Q{(d.month - 1) // 3 + 1}"
        if not quarters or quarters[-1] != q_key:
            quarters.append(q_key)
    quarters = sorted(list(set(quarters)))
    test_quarters = [q for q in quarters if q >= "2024Q1"]

    all_oos = []
    all_metrics = []

    total_folds = len(horizons) * len(test_quarters)
    fold_idx = 0
    for n in horizons:
        label_col = f"mfe_{n}_ge7"
        for test_q in test_quarters:
            fold_idx += 1
            test_year, test_qnum = int(test_q[:4]), int(test_q[-1])
            test_start = pd.Timestamp(f"{test_year}-{(test_qnum-1)*3+1:02d}-01")
            test_end_month = test_qnum * 3
            if test_end_month == 12:
                test_end = pd.Timestamp(f"{test_year}-12-31")
            else:
                test_end = pd.Timestamp(f"{test_year}-{test_end_month+1:02d}-01") - pd.Timedelta(days=1)

            # 标签裁剪：训练集截止提前 max(horizons) 天
            cutoff = _get_trailing_cutoff(all_dates, test_start, max(HORIZONS))

            # 计算训练集起始日期
            if window_quarters and window_quarters > 0:
                # 滚动窗口：从测试季度前 window_quarters 个季度开始
                q_idx = quarters.index(test_q) if test_q in quarters else -1
                if q_idx < 0:
                    continue
                train_start_q_idx = max(0, q_idx - window_quarters)
                train_start_q = quarters[train_start_q_idx]
                ts_year, ts_qnum = int(train_start_q[:4]), int(train_start_q[-1])
                train_start = pd.Timestamp(f"{ts_year}-{(ts_qnum-1)*3+1:02d}-01")
            else:
                # expanding：从数据最早日期开始
                train_start = data_start

            train_mask = (df["selection_date"] >= train_start) & (df["selection_date"] <= cutoff)
            test_mask = (df["selection_date"] >= test_start) & (df["selection_date"] <= test_end)

            # 只保留必要列，减少内存
            needed_cols = list(set(
                ["selection_date", "ts_code", "stock_name", "board",
                 f"mfe_{n}", f"mae_{n}", f"ret_{n}", label_col]
                + lag_cols
            ))
            available_cols = [c for c in needed_cols if c in df.columns]
            train_df = df.loc[train_mask, available_cols].copy()
            test_df = df.loc[test_mask, available_cols].copy()

            train_df[label_col] = (train_df[f"mfe_{n}"] >= PROFIT_THRESHOLD).astype(int)
            test_df[label_col] = (test_df[f"mfe_{n}"] >= PROFIT_THRESHOLD).astype(int)

            if len(train_df) < 200 or len(test_df) < 30:
                print(f"  [{fold_idx}/{total_folds}] {window_name} H={n} {test_q}: "
                      f"样本不足(train={len(train_df)})，跳过")
                continue

            # 计算时间加权
            sample_weights = None
            if use_weight:
                days_to_test = (test_start - train_df["selection_date"]).dt.days
                quarters_to_test = days_to_test / 90.0
                # 权重衰减：最近1Q=1.0, 2Q=0.8, 3-4Q=0.6, 5-8Q=0.3
                sample_weights = np.where(
                    quarters_to_test <= 1, 1.0,
                    np.where(quarters_to_test <= 2, 0.8,
                             np.where(quarters_to_test <= 4, 0.6, 0.3))
                ).astype(np.float64)

            metrics, oos = _train_cls_fold(
                train_df, test_df, lag_cols, label_col,
                sample_weights=sample_weights
            )
            if metrics is None:
                print(f"  [{fold_idx}/{total_folds}] {window_name} H={n} {test_q}: 训练失败，跳过")
                continue

            metrics["horizon"] = n
            metrics["train_period"] = f"{train_start.strftime('%Y-%m-%d')}~{cutoff.strftime('%Y-%m-%d')}"
            metrics["test_period"] = test_q
            metrics["window"] = window_name
            all_metrics.append(metrics)

            oos["horizon"] = n
            oos["fold"] = test_q
            all_oos.append(oos)

            print(f"  [{fold_idx}/{total_folds}] {window_name} H={n} {test_q}: "
                  f"AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
                  f"PosRate={metrics['pos_rate']:.3f} BestIter={metrics['best_iter']} "
                  f"TrainN={metrics['train_n']}")

    oos_df = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    return oos_df, all_metrics


# ===========================================================================
# 组合净值回测
# ===========================================================================
def portfolio_backtest(oos_df, horizon, top_n,
                       buy_cost, sell_cost, min_amount):
    """
    真实组合净值回测：
    - 每个交易日截面选 Top N
    - 过滤涨停、低成交额
    - 计算实际持有收益
    - 非重叠 NAV 曲线 + 重叠收益统计
    按需查询 kline 数据，避免预加载大对象。
    """
    if oos_df.empty:
        return None, None, None

    ret_col = f"ret_{horizon}"

    # 按需查询 kline 获取成交额和涨跌幅
    oos_df = oos_df.copy()
    oos_df["daily_amount"] = np.nan
    oos_df["pct_chg"] = np.nan

    engine = create_engine(DATABASE_URL)
    ts_codes = oos_df["ts_code"].unique().tolist()
    for i in range(0, len(ts_codes), 200):
        batch = ts_codes[i:i + 200]
        placeholders = ",".join([f"'{c}'" for c in batch])
        sql = (
            f"SELECT ts_code, bar_time, close, volume "
            f"FROM stock_k_data WHERE freq='d' AND ts_code IN ({placeholders}) "
            f"AND bar_time >= '2023-01-01' ORDER BY ts_code, bar_time"
        )
        with engine.connect() as conn:
            kline = pd.read_sql(sql, conn)
        if kline.empty:
            continue
        kline["bar_time"] = pd.to_datetime(kline["bar_time"]).dt.normalize()
        kline["amount"] = kline["close"] * kline["volume"]
        for ts_code, group in kline.groupby("ts_code"):
            g = group.set_index("bar_time").sort_index()
            prev_close = g["close"].shift(1)
            pct_chg = g["close"] / prev_close - 1
            mask = oos_df["ts_code"] == ts_code
            subset = oos_df[mask]
            common = subset["selection_date"][subset["selection_date"].isin(g.index)]
            if common.empty:
                continue
            oos_df.loc[common.index, "daily_amount"] = g.loc[common, "amount"].values
            oos_df.loc[common.index, "pct_chg"] = pct_chg.loc[common].values
        del kline

    # 向量化过滤：涨停 + 低成交额
    limit_pct = oos_df["ts_code"].apply(_get_limit_pct)
    limit_up_mask = oos_df["pct_chg"].notna() & (oos_df["pct_chg"] >= limit_pct - 0.005)
    low_amount_mask = oos_df["daily_amount"] < min_amount
    filter_mask = limit_up_mask | low_amount_mask
    oos_filtered = oos_df[~filter_mask].copy()
    n_filtered = filter_mask.sum()
    print(f"  H={horizon}: 过滤 {n_filtered} 条（涨停+低成交额），剩余 {len(oos_filtered)} 条")

    # 每日截面选 Top N
    dates = sorted(oos_filtered["selection_date"].unique())
    daily_portfolios = []
    for dt in dates:
        day_data = oos_filtered[oos_filtered["selection_date"] == dt]
        day_data = day_data.dropna(subset=["pred_prob", ret_col])
        if len(day_data) < 5:
            continue
        top = day_data.nlargest(top_n, "pred_prob")
        daily_portfolios.append(top)

    if not daily_portfolios:
        return None, None, None

    port_df = pd.concat(daily_portfolios, ignore_index=True)

    # 重叠收益统计
    overlap_stats = _compute_overlap_stats(port_df, dates, horizon, ret_col, buy_cost, sell_cost)

    # 非重叠 NAV 曲线
    nav_df = _compute_nav_curve(port_df, dates, horizon, ret_col, buy_cost, sell_cost)

    # 持仓明细
    holdings = port_df[["selection_date", "ts_code", "stock_name", "pred_prob",
                        ret_col, "mae_" + str(horizon), "mfe_" + str(horizon),
                        "pct_chg", "daily_amount", "board", "horizon"]].copy()
    holdings["rank"] = holdings.groupby("selection_date")["pred_prob"].rank(ascending=False, method="first")

    return nav_df, overlap_stats, holdings


def _compute_overlap_stats(port_df, dates, horizon, ret_col, buy_cost, sell_cost):
    """重叠收益统计"""
    daily_rets = []
    for dt in dates:
        day = port_df[port_df["selection_date"] == dt]
        if len(day) < 3:
            continue
        net_rets = day[ret_col] - buy_cost - sell_cost
        daily_rets.append({
            "date": dt,
            "avg_ret": net_rets.mean(),
            "median_ret": net_rets.median(),
            "hit_rate": (net_rets > 0).mean(),
            "n_stocks": len(day),
        })

    if not daily_rets:
        return {}

    ret_series = pd.DataFrame(daily_rets)
    avg_rets = ret_series["avg_ret"]

    # 重叠截面夏普未做自相关调整，仅供参考；以非重叠 NAV 夏普为准
    eq_daily = (1 + avg_rets) ** (1 / horizon) - 1
    ann_ret = (1 + eq_daily.mean()) ** 242 - 1
    ann_vol = eq_daily.std() * np.sqrt(242)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # 累积收益最大回撤
    cum = (1 + avg_rets).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1
    max_dd = dd.min()

    return {
        "horizon": horizon,
        "n_days": len(ret_series),
        "avg_n_day_ret": round(avg_rets.mean(), 4),
        "hit_rate": round(ret_series["hit_rate"].mean(), 4),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "avg_n_stocks": round(ret_series["n_stocks"].mean(), 1),
    }


def _compute_nav_curve(port_df, dates, horizon, ret_col, buy_cost, sell_cost):
    """非重叠 NAV 曲线"""
    rebalance_dates = dates[::horizon]
    nav_records = []
    nav = 1.0

    for rb_date in rebalance_dates:
        day = port_df[port_df["selection_date"] == rb_date]
        if len(day) < 3:
            continue
        net_rets = day[ret_col] - buy_cost - sell_cost
        port_ret = net_rets.mean()
        nav *= (1 + port_ret)
        nav_records.append({
            "date": rb_date,
            "nav": round(nav, 4),
            "period_ret": round(port_ret, 4),
            "n_stocks": len(day),
        })

    if not nav_records:
        return pd.DataFrame()

    nav_df = pd.DataFrame(nav_records)

    total_ret = nav_df["nav"].iloc[-1] - 1
    n_periods = len(nav_df)
    total_days = n_periods * horizon
    ann_ret = (1 + total_ret) ** (242 / max(total_days, 1)) - 1

    running_max = nav_df["nav"].cummax()
    drawdown = nav_df["nav"] / running_max - 1
    max_dd = drawdown.min()
    win_rate = (nav_df["period_ret"] > 0).mean()

    # 直接用周期收益率计算夏普，避免日化转换引入偏差
    period_rets = nav_df["period_ret"]
    n_periods_per_year = 242 / horizon
    ann_vol = period_rets.std() * np.sqrt(n_periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    print(f"  H={horizon} NAV: 总收益={total_ret:.4f} 年化={ann_ret:.4f} "
          f"夏普={sharpe:.4f} 最大回撤={max_dd:.4f} 胜率={win_rate:.4f} "
          f"调仓{n_periods}次")

    return nav_df


# ===========================================================================
# 诊断函数
# ===========================================================================
def _compute_psi(expected, actual, n_bins=10):
    """计算 PSI（群体稳定性指数）

    衡量两个分布的漂移程度：
        <0.1 稳定, 0.1-0.2 轻微漂移, >0.2 显著漂移

    Args:
        expected: 基准期特征值（np.array）
        actual:   观测期特征值（np.array）
        n_bins:   分桶数
    Returns:
        PSI 值
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    # 用基准期分位数作为分桶边界
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    # 去重：避免空桶
    bins = np.unique(bins)

    expected_counts = np.histogram(expected, bins=bins)[0].astype(float)
    actual_counts = np.histogram(actual, bins=bins)[0].astype(float)

    # 避免除零：空桶加 0.0001
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def _diagnose_recent_failure(df, oos_df, feature_cols, test_quarter="2026Q2"):
    """诊断近期模型失效原因

    检查项：
    1. 特征分布漂移 PSI（基准期 vs 观测期）
    2. 标签正样本率变化
    3. Top 组板块分布
    4. 分数分层收益（排序是否反向）

    Args:
        df:          完整特征 DataFrame（含 selection_date, ts_code, board, mfe_N）
        oos_df:      OOS 预测结果（含 pred_prob, horizon, fold）
        feature_cols:特征列名列表
        test_quarter:要诊断的季度，如 "2026Q2"
    Returns:
        dict 诊断结果
    """
    df["selection_date"] = pd.to_datetime(df["selection_date"]).dt.normalize()

    # 解析季度
    test_year, test_qnum = int(test_quarter[:4]), int(test_quarter[-1])
    test_start = pd.Timestamp(f"{test_year}-{(test_qnum-1)*3+1:02d}-01")
    if test_qnum == 4:
        test_end = pd.Timestamp(f"{test_year}-12-31")
    else:
        test_end = pd.Timestamp(f"{test_year}-{test_qnum*3+1:02d}-01") - pd.Timedelta(days=1)

    # 基准期：前一个季度
    base_qnum = test_qnum - 1 if test_qnum > 1 else 4
    base_year = test_year if test_qnum > 1 else test_year - 1
    base_start = pd.Timestamp(f"{base_year}-{(base_qnum-1)*3+1:02d}-01")
    if base_qnum == 4:
        base_end = pd.Timestamp(f"{base_year}-12-31")
    else:
        base_end = pd.Timestamp(f"{base_year}-{base_qnum*3+1:02d}-01") - pd.Timedelta(days=1)

    base_mask = (df["selection_date"] >= base_start) & (df["selection_date"] <= base_end)
    test_mask = (df["selection_date"] >= test_start) & (df["selection_date"] <= test_end)
    base_df = df[base_mask]
    test_df_full = df[test_mask]

    results = {"test_quarter": test_quarter, "base_quarter": f"{base_year}Q{base_qnum}"}

    # ---- 1. 特征分布漂移 PSI ----
    psi_results = {}
    for col in feature_cols:
        if col in base_df.columns and col in test_df_full.columns:
            base_vals = base_df[col].dropna().values
            test_vals = test_df_full[col].dropna().values
            if len(base_vals) > 50 and len(test_vals) > 50:
                psi_val = _compute_psi(base_vals, test_vals)
                psi_results[col] = round(psi_val, 4)

    # 按 PSI 降序排列，取 Top 10
    psi_sorted = sorted(psi_results.items(), key=lambda x: x[1], reverse=True)
    results["psi_top10"] = psi_sorted[:10]
    n_significant = sum(1 for _, v in psi_results.items() if v > 0.2)
    n_mild = sum(1 for _, v in psi_results.items() if 0.1 < v <= 0.2)
    results["psi_summary"] = {
        "n_features": len(psi_results),
        "n_significant_drift": n_significant,
        "n_mild_drift": n_mild,
    }

    # ---- 2. 标签正样本率变化 ----
    pos_rate_by_quarter = {}
    for n in HORIZONS:
        mfe_col = f"mfe_{n}"
        if mfe_col not in df.columns:
            continue
        base_pos = (base_df[mfe_col] >= PROFIT_THRESHOLD).mean()
        test_pos = (test_df_full[mfe_col] >= PROFIT_THRESHOLD).mean()
        pos_rate_by_quarter[n] = {
            "base": round(base_pos, 4),
            "test": round(test_pos, 4),
            "delta": round(test_pos - base_pos, 4),
        }
    results["pos_rate_change"] = pos_rate_by_quarter

    # ---- 3. Top 组板块分布 ----
    board_dist = {}
    for n in HORIZONS:
        oos_h = oos_df[(oos_df["horizon"] == n) & (oos_df["fold"] == test_quarter)]
        if oos_h.empty:
            continue
        # Top 10%
        threshold = oos_h["pred_prob"].quantile(0.9)
        top = oos_h[oos_h["pred_prob"] >= threshold]
        if "board" in top.columns and not top.empty:
            board_counts = top["board"].value_counts(normalize=True)
            board_dist[n] = board_counts.round(3).to_dict()
    results["top_board_dist"] = board_dist

    # ---- 4. 分数分层收益（排序是否反向）----
    score_layer_ret = {}
    for n in HORIZONS:
        oos_h = oos_df[(oos_df["horizon"] == n) & (oos_df["fold"] == test_quarter)]
        if oos_h.empty:
            continue
        mfe_col = f"mfe_{n}"
        if mfe_col not in oos_h.columns:
            continue
        oos_h = oos_h.dropna(subset=["pred_prob", mfe_col])
        if len(oos_h) < 50:
            continue
        oos_h["score_bin"] = pd.qcut(oos_h["pred_prob"], 5, labels=False, duplicates="drop")
        layer_ret = oos_h.groupby("score_bin")[mfe_col].mean()
        score_layer_ret[n] = layer_ret.round(4).to_dict()
    results["score_layer_mfe"] = score_layer_ret

    return results


def _print_diagnosis(diagnosis):
    """打印诊断结果"""
    tq = diagnosis["test_quarter"]
    bq = diagnosis["base_quarter"]
    print(f"\n  === {tq} 失效诊断（基准期: {bq}）===")

    # PSI
    psi_summary = diagnosis.get("psi_summary", {})
    print(f"\n  [PSI 特征漂移] 共 {psi_summary.get('n_features', 0)} 个特征, "
          f"显著漂移(>0.2): {psi_summary.get('n_significant_drift', 0)}, "
          f"轻微漂移(0.1-0.2): {psi_summary.get('n_mild_drift', 0)}")
    psi_top = diagnosis.get("psi_top10", [])
    if psi_top:
        print("  Top10 漂移特征:")
        for feat, psi_val in psi_top:
            flag = "***" if psi_val > 0.2 else ("**" if psi_val > 0.1 else "")
            print(f"    {feat:30s} PSI={psi_val:.4f} {flag}")

    # 正样本率
    pos_rate = diagnosis.get("pos_rate_change", {})
    if pos_rate:
        print(f"\n  [标签正样本率变化] MFE≥{PROFIT_THRESHOLD:.0%}")
        for n, v in pos_rate.items():
            delta_str = f"+{v['delta']:.4f}" if v['delta'] >= 0 else f"{v['delta']:.4f}"
            print(f"    H={n:2d}: {bq}={v['base']:.4f} → {tq}={v['test']:.4f} ({delta_str})")

    # 板块分布
    board_dist = diagnosis.get("top_board_dist", {})
    if board_dist:
        print(f"\n  [Top10% 板块分布]")
        for n, dist in board_dist.items():
            dist_str = "  ".join([f"{k}={v:.1%}" for k, v in sorted(dist.items(), key=lambda x: -x[1])])
            print(f"    H={n:2d}: {dist_str}")

    # 分数分层收益
    score_layer = diagnosis.get("score_layer_mfe", {})
    if score_layer:
        print(f"\n  [分数分层 MFE] (0=最低分, 4=最高分)")
        for n, layers in score_layer.items():
            layer_str = "  ".join([f"Bin{k}={v:.4f}" for k, v in sorted(layers.items())])
            # 检查是否反向：高分组的 MFE 应该更高
            sorted_vals = [layers.get(i, 0) for i in sorted(layers.keys())]
            is_reversed = len(sorted_vals) >= 2 and sorted_vals[-1] < sorted_vals[0]
            flag = " ← 反向!" if is_reversed else ""
            print(f"    H={n:2d}: {layer_str}{flag}")


# ===========================================================================
# 持仓集中度分析
# ===========================================================================
def analyze_holdings(holdings_df, top_n):
    """持仓集中度分析"""
    if holdings_df is None or holdings_df.empty:
        return {}

    results = {}

    # 年份分布
    holdings_df["year"] = pd.to_datetime(holdings_df["selection_date"]).dt.year
    year_dist = holdings_df.groupby("year").agg(
        n_records=("ts_code", "count"),
        n_unique_stocks=("ts_code", "nunique"),
    )
    results["year_distribution"] = year_dist.to_dict("index")

    # 个股重复度 Top 20
    stock_freq = holdings_df.groupby("ts_code").agg(
        count=("ts_code", "size"),
        stock_name=("stock_name", "first"),
    ).sort_values("count", ascending=False).head(20)

    # 板块分布
    board_dist = holdings_df["board"].value_counts().to_dict()
    results["board_distribution"] = board_dist

    # 成交额分布
    if "daily_amount" in holdings_df.columns:
        amt = holdings_df["daily_amount"].dropna()
        if len(amt) > 0:
            results["amount_stats"] = {
                "mean_M": round(amt.mean() / 1e6, 0),
                "median_M": round(amt.median() / 1e6, 0),
                "pct_below_50M": round((amt < 50_000_000).mean(), 4),
            }

    # 打印
    print("\n  ===== 持仓集中度分析 =====")
    print(f"  总持仓记录: {len(holdings_df)}")
    print(f"  唯一股票数: {holdings_df['ts_code'].nunique()}")
    print(f"  板块分布: {board_dist}")
    if "amount_stats" in results:
        a = results["amount_stats"]
        print(f"  成交额: 均值={a['mean_M']:.0f}M 中位数={a['median_M']:.0f}M "
              f"<50M占比={a['pct_below_50M']:.1%}")
    print(f"  Top 10 高频股:")
    for i, (ts_code, row) in enumerate(stock_freq.head(10).iterrows()):
        print(f"    {i+1}. {ts_code} {row['stock_name']} 出现{row['count']}次")

    return results


# ===========================================================================
# 单日预测（lag5_rolling_4q）
# ===========================================================================
def predict_for_date(predict_date: str, top_n: int = TOP_N,
                     min_amount: float = MIN_AMOUNT):
    """使用 lag5_rolling_4q 配置训练最终模型，预测指定日期的选股结果

    训练窗口与回测一致：预测日期所在季度的前 4 个季度，
    cutoff 提前 max(HORIZONS) 个交易日防止标签穿越。

    Args:
        predict_date: 预测日期，格式 YYYY-MM-DD
        top_n: 输出 Top N 股票
        min_amount: 最低日成交额过滤阈值
    """
    predict_ts = pd.Timestamp(predict_date)

    # 确定预测日期所在季度
    predict_q = f"{predict_ts.year}Q{(predict_ts.month - 1) // 3 + 1}"
    print(f"预测日期: {predict_date} (季度: {predict_q})")
    print(f"配置: lag5_rolling_4q, HORIZONS={HORIZONS}, TopN={top_n}")

    # [1/5] 加载数据
    print("\n[1/5] 加载数据 ...")
    sel_df = load_data()
    if sel_df.empty:
        print("  无数据，退出")
        return

    # 检查预测日期是否有数据
    sel_df["selection_date"] = pd.to_datetime(sel_df["selection_date"]).dt.normalize()
    available_dates = sorted(sel_df["selection_date"].unique())
    if predict_ts not in available_dates:
        # 找最近的日期
        nearest = min(available_dates, key=lambda d: abs((d - predict_ts).days))
        if abs((nearest - predict_ts).days) <= 3:
            print(f"  预测日期 {predict_date} 无数据，使用最近日期 {nearest.strftime('%Y-%m-%d')}")
            predict_ts = nearest
        else:
            print(f"  预测日期 {predict_date} 无数据，最近日期 {nearest.strftime('%Y-%m-%d')} 差距过大，退出")
            return

    ts_codes = sel_df["ts_code"].unique().tolist()

    # [2/5] 预加载 kline + 计算未来收益
    print("\n[2/5] 预加载 kline ...")
    kline_dict = preload_kline(ts_codes)

    print("\n[3/5] 计算未来收益（MFE/MAE/RET）...")
    df = compute_future_returns(kline_dict, sel_df, HORIZONS)
    del kline_dict
    del sel_df
    import gc; gc.collect()

    # [4/5] 构建 lag5 特征
    print("\n[4/5] 构建连续 5 bar 特征 ...")
    df_lag, lag_feature_cols = build_lag_features(df, n_lags=5)
    df = pd.concat([df, df_lag], axis=1)
    del df_lag
    gc.collect()

    # 释放不需要的列
    keep_cols = {"selection_date", "ts_code", "stock_name", "board"} | set(lag_feature_cols)
    for n in HORIZONS:
        keep_cols |= {f"ret_{n}", f"mae_{n}", f"mfe_{n}"}
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        gc.collect()

    # [5/5] 训练模型 + 预测
    print("\n[5/5] 训练模型 + 预测 ...")

    # 提取季度列表（与 walk_forward_rolling 一致）
    all_dates = sorted(df["selection_date"].unique())
    quarters = []
    for d in all_dates:
        q_key = f"{d.year}Q{(d.month - 1) // 3 + 1}"
        if not quarters or quarters[-1] != q_key:
            quarters.append(q_key)
    quarters = sorted(list(set(quarters)))

    # 确定 test_quarter 和训练窗口
    if predict_q not in quarters:
        print(f"  季度 {predict_q} 不在数据中，可用季度: {quarters}")
        return

    q_idx = quarters.index(predict_q)
    window_quarters = 4
    train_start_q_idx = max(0, q_idx - window_quarters)
    train_start_q = quarters[train_start_q_idx]

    test_year, test_qnum = int(predict_q[:4]), int(predict_q[-1])
    test_start = pd.Timestamp(f"{test_year}-{(test_qnum-1)*3+1:02d}-01")

    ts_year, ts_qnum = int(train_start_q[:4]), int(train_start_q[-1])
    train_start = pd.Timestamp(f"{ts_year}-{(ts_qnum-1)*3+1:02d}-01")

    # 标签穿越防护 cutoff
    cutoff = _get_trailing_cutoff(all_dates, test_start, max(HORIZONS))

    print(f"  训练窗口: {train_start.strftime('%Y-%m-%d')} ~ {cutoff.strftime('%Y-%m-%d')}")
    print(f"  预测日期: {predict_ts.strftime('%Y-%m-%d')}")

    # 训练集 + 预测集
    train_mask = (df["selection_date"] >= train_start) & (df["selection_date"] <= cutoff)
    pred_mask = df["selection_date"] == predict_ts

    # 过滤成交额（预测集）
    # 注意：dsa_selection.avg_amount_20d 单位是亿，MIN_AMOUNT 单位是元
    if "avg_amount_20d" not in df.columns:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            amt_df = pd.read_sql(
                f"SELECT ts_code, avg_amount_20d FROM dsa_selection "
                f"WHERE selection_date = '{predict_ts.strftime('%Y-%m-%d')}'",
                conn
            )
        if not amt_df.empty:
            pred_ts_codes = df.loc[pred_mask, "ts_code"].values
            amt_map = dict(zip(amt_df["ts_code"], amt_df["avg_amount_20d"]))
            # avg_amount_20d 单位是亿，转换为元再比较
            min_amount_yi = min_amount / 1e8
            low_amt_stocks = [c for c in pred_ts_codes
                              if (amt_map.get(c, 0) or 0) < min_amount_yi]
            if low_amt_stocks:
                print(f"  过滤低成交额股票: {len(low_amt_stocks)} 只 (<{min_amount_yi:.1f}亿)")
                pred_mask = pred_mask & ~df["ts_code"].isin(low_amt_stocks)

    models = {}
    predictions = {}

    for n in HORIZONS:
        label_col = f"mfe_{n}_ge7"
        print(f"\n  --- H={n} ---")

        needed_cols = list(set(
            ["selection_date", "ts_code", "stock_name", "board",
             f"mfe_{n}", f"mae_{n}", f"ret_{n}", label_col]
            + lag_feature_cols
        ))
        available_cols = [c for c in needed_cols if c in df.columns]

        train_df = df.loc[train_mask, available_cols].copy()
        pred_df = df.loc[pred_mask, available_cols].copy()

        # 构造标签
        train_df[label_col] = (train_df[f"mfe_{n}"] >= PROFIT_THRESHOLD).astype(int)
        # 预测集不需要标签，但 _train_cls_fold 需要 label_col 列存在
        pred_df[label_col] = 0  # 占位，不影响预测

        if len(train_df) < 200:
            print(f"    训练样本不足 ({len(train_df)})，跳过")
            continue
        if len(pred_df) < 1:
            print(f"    预测样本为空，跳过")
            continue

        result = _train_cls_fold(
            train_df, pred_df, lag_feature_cols, label_col,
            return_model=True,
        )

        if result[0] is None:
            print(f"    训练失败，跳过")
            continue

        metrics, oos, model = result
        models[n] = model

        print(f"    训练 AUC={metrics['auc']:.4f} BestIter={metrics['best_iter']} "
              f"TrainN={metrics['train_n']} PosRate={metrics['pos_rate']:.3f}")

        # 保存模型
        model_dir = os.path.join(OUTPUT_DIR, "dsa_gbdt")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_h{n}.txt")
        model.booster_.save_model(model_path)
        print(f"    模型已保存: {model_path}")

        # 预测结果
        pred_df_result = pred_df[["ts_code", "stock_name", "board"]].copy()
        pred_df_result["pred_prob"] = oos["pred_prob"].values
        predictions[n] = pred_df_result

        # 打印 Top N
        top = pred_df_result.nlargest(top_n, "pred_prob")
        print(f"\n    Top {top_n} (H={n}):")
        for i, (_, row) in enumerate(top.iterrows()):
            print(f"      {i+1:2d}. {row['ts_code']} {row['stock_name']:8s} "
                  f"[{row['board']}] prob={row['pred_prob']:.4f}")

    # 合并 H=5 和 H=10 的预测结果
    if not predictions:
        print("\n无有效预测结果")
        return

    # 以 H=5 为主表，合并 H=10
    result_dfs = []
    for n in HORIZONS:
        if n not in predictions:
            continue
        pdf = predictions[n][["ts_code", "stock_name", "board", "pred_prob"]].copy()
        pdf = pdf.rename(columns={"pred_prob": f"pred_prob_h{n}"})
        pdf[f"rank_h{n}"] = pdf[f"pred_prob_h{n}"].rank(ascending=False, method="first").astype(int)
        result_dfs.append(pdf)

    if len(result_dfs) == 2:
        merged = result_dfs[0].merge(
            result_dfs[1][["ts_code", "pred_prob_h10", "rank_h10"]],
            on="ts_code", how="outer"
        )
    else:
        merged = result_dfs[0]

    # 按 H=5 概率降序排列
    sort_col = "pred_prob_h5" if "pred_prob_h5" in merged.columns else merged.columns[3]
    merged = merged.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, "dsa_gbdt", f"predict_{predict_ts.strftime('%Y%m%d')}.csv")
    merged.to_csv(csv_path, index=False)
    print(f"\n预测结果已保存: {csv_path}")
    print(f"共 {len(merged)} 只股票")

    # 综合排名打印
    print(f"\n{'=' * 80}")
    print(f"预测日期: {predict_ts.strftime('%Y-%m-%d')} | lag5_rolling_4q")
    print(f"{'=' * 80}")
    top_merged = merged.head(top_n)
    print(f"\nTop {top_n} 综合排名:")
    for i, (_, row) in enumerate(top_merged.iterrows()):
        h5_prob = row.get('pred_prob_h5', float('nan'))
        h5_rank = row.get('rank_h5', '-')
        h10_prob = row.get('pred_prob_h10', float('nan'))
        h10_rank = row.get('rank_h10', '-')
        print(f"  {i+1:2d}. {row['ts_code']} {row['stock_name']:8s} [{row['board']}] "
              f"H5={h5_prob:.4f}(#{h5_rank}) H10={h10_prob:.4f}(#{h10_rank})")

    del df
    gc.collect()


# ===========================================================================
# 主函数：N_LAGS × 训练窗口 交叉实验
# ===========================================================================
WINDOW_CONFIGS = [
    ("expanding",    None, False),
    ("rolling_2q",   2,    False),
    ("rolling_4q",   4,    False),
    ("rolling_6q",   6,    False),
    ("rolling_8q",   8,    False),
    ("weighted_8q",  8,    True),
]

# N_LAGS=10 特征数翻倍（180），expanding 窗口训练集过大（46万行）会 OOM，
# 因此 lag10 只跑 rolling 窗口
LAG_WINDOW_FILTERS = {
    5: None,  # None = 所有窗口
    10: {"rolling_2q", "rolling_4q", "rolling_6q"},  # 只跑 rolling 窗口
}

LAG_CONFIGS = [5, 10]


def main():
    parser = argparse.ArgumentParser(description="DSA 选股 GBDT N_LAGS × 训练窗口实验")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="每期持仓数")
    parser.add_argument("--min-amount", type=float, default=MIN_AMOUNT, help="最低日成交额")
    parser.add_argument("--windows", type=str, default=None,
                        help="只跑指定窗口，逗号分隔，如 rolling_4q,expanding")
    parser.add_argument("--n-lags", type=str, default=None,
                        help="只跑指定 N_LAGS，逗号分隔，如 5,10")
    parser.add_argument("--predict-date", type=str, default=None,
                        help="单日预测模式，指定日期如 2026-06-02，使用 lag5_rolling_4q 配置")
    args = parser.parse_args()

    # 单日预测模式
    if args.predict_date:
        predict_for_date(args.predict_date, top_n=args.top_n, min_amount=args.min_amount)
        return

    top_n = args.top_n
    min_amount = args.min_amount

    # 过滤窗口
    if args.windows:
        selected = set(args.windows.split(","))
        window_configs = [(n, wq, uw) for n, wq, uw in WINDOW_CONFIGS if n in selected]
    else:
        window_configs = WINDOW_CONFIGS

    # 过滤 N_LAGS
    if args.n_lags:
        lag_configs = [int(x) for x in args.n_lags.split(",")]
    else:
        lag_configs = LAG_CONFIGS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("DSA 选股 GBDT N_LAGS × 训练窗口实验")
    print(f"  标签: MFE≥{PROFIT_THRESHOLD:.0%}")
    print(f"  Horizons: {HORIZONS}")
    print(f"  N_LAGS: {lag_configs}")
    print(f"  TopN={top_n} MinAmount={min_amount/1e6:.0f}M")
    print(f"  买入成本={BUY_COST:.2%} 卖出成本={SELL_COST:.2%}")
    print(f"  训练窗口: {[n for n, _, _ in window_configs]}")
    print("=" * 80)

    # [1/4] 加载数据
    print("\n[1/4] 加载数据 ...")
    sel_df = load_data()
    if sel_df.empty:
        print("  无数据，退出")
        return

    ts_codes = sel_df["ts_code"].unique().tolist()

    # [2/4] 预加载 kline + 计算未来收益
    print("\n[2/4] 预加载 kline ...")
    kline_dict = preload_kline(ts_codes)

    print("\n[3/4] 计算未来收益（MFE/MAE/RET）...")
    df = compute_future_returns(kline_dict, sel_df, HORIZONS)

    del kline_dict
    del sel_df
    import gc; gc.collect()

    # [4/4] N_LAGS × 窗口 交叉实验
    all_results = {}  # key: "lag5_expanding" 格式

    for n_lags in lag_configs:
        print(f"\n{'=' * 80}")
        print(f"构造连续 {n_lags} bar 特征 ...")
        print(f"{'=' * 80}")

        df_lag, lag_feature_cols = build_lag_features(df, n_lags=n_lags)
        # 将 lag 列合并到 df，然后释放临时 lag 列
        df_with_lag = pd.concat([df, df_lag], axis=1)
        del df_lag
        gc.collect()

        # 释放不需要的列，减少内存
        keep_cols = {"selection_date", "ts_code", "stock_name", "board"} | set(lag_feature_cols)
        for n in HORIZONS:
            keep_cols |= {f"ret_{n}", f"mae_{n}", f"mfe_{n}"}
        drop_cols = [c for c in df_with_lag.columns if c not in keep_cols]
        if drop_cols:
            df_with_lag.drop(columns=drop_cols, inplace=True)
            gc.collect()

        for window_name, window_quarters, use_weight in window_configs:
            # lag10 只跑指定窗口，避免 OOM
            allowed = LAG_WINDOW_FILTERS.get(n_lags)
            if allowed is not None and window_name not in allowed:
                print(f"  跳过 lag{n_lags}_{window_name}（N_LAGS={n_lags} 不支持此窗口）")
                continue

            exp_name = f"lag{n_lags}_{window_name}"
            print(f"\n{'=' * 80}")
            print(f"实验: {exp_name} (N_LAGS={n_lags}, 窗口={window_name})")
            print(f"{'=' * 80}")

            oos_df, metrics_list = walk_forward_rolling(
                df_with_lag, lag_feature_cols, HORIZONS,
                window_quarters=window_quarters,
                use_weight=use_weight,
                window_name=exp_name,
            )

            if oos_df.empty:
                print(f"  {exp_name}: 无 OOS 预测，跳过组合回测")
                all_results[exp_name] = {"metrics": metrics_list, "nav": None, "n_lags": n_lags}
                continue

            # 组合回测
            all_nav = []
            all_overlap_stats = []

            for n in HORIZONS:
                oos_h = oos_df[oos_df["horizon"] == n].copy()
                if oos_h.empty:
                    continue

                nav_df, overlap_stats, _ = portfolio_backtest(
                    oos_h, n, top_n, BUY_COST, SELL_COST, min_amount
                )
                if nav_df is not None and not nav_df.empty:
                    nav_df["horizon"] = n
                    all_nav.append(nav_df)
                if overlap_stats:
                    overlap_stats["window"] = exp_name
                    all_overlap_stats.append(overlap_stats)

            # 保存结果
            if metrics_list:
                pd.DataFrame(metrics_list).to_csv(
                    os.path.join(OUTPUT_DIR, f"dsa_gbdt_backtest_wf_{exp_name}.csv"),
                    index=False)
                print(f"  {exp_name} WF 指标已保存")

            if all_nav:
                nav_all = pd.concat(all_nav, ignore_index=True)
                nav_all["window"] = exp_name
                nav_all.to_csv(
                    os.path.join(OUTPUT_DIR, f"dsa_gbdt_backtest_nav_{exp_name}.csv"),
                    index=False)
                print(f"  {exp_name} NAV 已保存")

            # 2026Q2 诊断
            q2_data = oos_df[oos_df["fold"] == "2026Q2"] if "fold" in oos_df.columns else pd.DataFrame()
            if not q2_data.empty:
                print(f"\n--- {exp_name} 2026Q2 失效诊断 ---")
                diagnosis = _diagnose_recent_failure(df_with_lag, oos_df, lag_feature_cols, "2026Q2")
                _print_diagnosis(diagnosis)

            all_results[exp_name] = {
                "metrics": metrics_list,
                "overlap_stats": all_overlap_stats,
                "nav": all_nav if all_nav else None,
                "n_lags": n_lags,
            }

            del oos_df
            gc.collect()

        del df_with_lag
        gc.collect()

    # ===== 实验对比汇总 =====
    print(f"\n{'=' * 80}")
    print("N_LAGS × 训练窗口 实验对比汇总")
    print(f"{'=' * 80}")

    # 1. 分季度 AUC 对比（H=5）
    print("\n--- 分季度 AUC 对比（H=5）---")
    for exp_name, results in all_results.items():
        metrics = results["metrics"]
        n_lags = results.get("n_lags", "?")
        if not metrics:
            continue
        h5_metrics = [m for m in metrics if m["horizon"] == 5]
        if not h5_metrics:
            continue
        auc_str = "  ".join([f"{m['test_period']}={m['auc']:.3f}" for m in h5_metrics])
        recent = h5_metrics[-4:]
        avg_auc = np.mean([m["auc"] for m in recent])
        print(f"  {exp_name:25s} (lag={n_lags}): {auc_str}  近4Q均值={avg_auc:.3f}")

    # 2. 分季度 AUC 对比（H=10）
    print("\n--- 分季度 AUC 对比（H=10）---")
    for exp_name, results in all_results.items():
        metrics = results["metrics"]
        n_lags = results.get("n_lags", "?")
        if not metrics:
            continue
        h10_metrics = [m for m in metrics if m["horizon"] == 10]
        if not h10_metrics:
            continue
        auc_str = "  ".join([f"{m['test_period']}={m['auc']:.3f}" for m in h10_metrics])
        recent = h10_metrics[-4:]
        avg_auc = np.mean([m["auc"] for m in recent])
        print(f"  {exp_name:25s} (lag={n_lags}): {auc_str}  近4Q均值={avg_auc:.3f}")

    # 3. 2026Q2 AUC 对比
    print("\n--- 2026Q2 AUC 对比 ---")
    for exp_name, results in all_results.items():
        metrics = results["metrics"]
        n_lags = results.get("n_lags", "?")
        if not metrics:
            continue
        q2_metrics = [m for m in metrics if m["test_period"] == "2026Q2"]
        if not q2_metrics:
            continue
        for m in q2_metrics:
            print(f"  {exp_name:25s} (lag={n_lags}) H={m['horizon']}: "
                  f"AUC={m['auc']:.4f} F1={m['f1']:.4f} TrainN={m['train_n']}")

    # 4. 非重叠 NAV 统计对比
    print("\n--- 非重叠 NAV 统计对比 ---")
    for exp_name, results in all_results.items():
        nav_list = results.get("nav")
        n_lags = results.get("n_lags", "?")
        if not nav_list:
            continue
        nav_all = pd.concat(nav_list, ignore_index=True) if isinstance(nav_list, list) else nav_list
        for n in HORIZONS:
            nav_h = nav_all[nav_all["horizon"] == n] if "horizon" in nav_all.columns else nav_all
            if nav_h.empty:
                continue
            total_ret = nav_h["nav"].iloc[-1] - 1
            n_periods = len(nav_h)
            total_days = n_periods * n
            ann_ret = (1 + total_ret) ** (242 / max(total_days, 1)) - 1
            period_rets = nav_h["period_ret"]
            n_periods_per_year = 242 / n
            ann_vol = period_rets.std() * np.sqrt(n_periods_per_year)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            running_max = nav_h["nav"].cummax()
            max_dd = (nav_h["nav"] / running_max - 1).min()
            win_rate = (period_rets > 0).mean()
            print(f"  {exp_name:25s} (lag={n_lags}) H={n}天: "
                  f"年化={ann_ret:.4f} 夏普={sharpe:.4f} "
                  f"最大回撤={max_dd:.4f} 胜率={win_rate:.4f} 调仓{n_periods}次")

    # 5. N_LAGS 对比总结
    print("\n--- N_LAGS 对比总结 ---")
    for n_lags in lag_configs:
        lag_exps = {k: v for k, v in all_results.items() if v.get("n_lags") == n_lags}
        if not lag_exps:
            continue
        # 取 rolling_4q 或第一个窗口的 AUC
        for exp_name, results in lag_exps.items():
            metrics = results["metrics"]
            if not metrics:
                continue
            h5 = [m for m in metrics if m["horizon"] == 5]
            h10 = [m for m in metrics if m["horizon"] == 10]
            avg5 = np.mean([m["auc"] for m in h5[-4:]]) if len(h5) >= 4 else 0
            avg10 = np.mean([m["auc"] for m in h10[-4:]]) if len(h10) >= 4 else 0
            q2_5 = next((m["auc"] for m in h5 if m["test_period"] == "2026Q2"), 0)
            q2_10 = next((m["auc"] for m in h10 if m["test_period"] == "2026Q2"), 0)
            print(f"  lag={n_lags} {exp_name}: 近4Q_AUC H5={avg5:.3f} H10={avg10:.3f} | "
                  f"2026Q2 H5={q2_5:.4f} H10={q2_10:.4f}")

    print(f"\n{'=' * 80}")
    print("实验完成")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
