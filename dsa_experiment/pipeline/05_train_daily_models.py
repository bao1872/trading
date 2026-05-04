#!/usr/bin/env python3
"""
日线买卖点模型训练：机会模型 + 风险模型

Purpose: 基于日线因子表训练 LightGBM 模型，预测未来5日可交易收益率
Inputs: output/daily_factor_table.parquet
Outputs: output/daily_return_model/, output/daily_risk_model/, output/daily_factor_with_scores.parquet
How to Run:
    python dsa_experiment/pipeline/05_train_daily_models.py
    python dsa_experiment/pipeline/05_train_daily_models.py --sample-limit 10000
    python dsa_experiment/pipeline/05_train_daily_models.py --strict-oos
Examples:
    python dsa_experiment/pipeline/05_train_daily_models.py
    python dsa_experiment/pipeline/05_train_daily_models.py --sample-limit 10000
    python dsa_experiment/pipeline/05_train_daily_models.py --strict-oos
Side Effects: 只读操作，输出模型文件和预测结果parquet

管线位置: Step 5/7 — 日线模型训练（机会模型+风险模型）

模型说明：
  机会模型：预测 ret_5_close_to_close（未来5日收盘收益率），找最优买点
  风险模型：预测 mae_5（未来5日最大回撤），做风控/卖点判断
  两个模型均使用 LightGBM 回归，3折滚动时间序列切分 + 25日embargo
  标签口径与周线一致：可交易的实际收益/风险

严格OOS模式（--strict-oos）：
  只保留最后一折的测试集预测作为真正样本外结果
  阈值（分位数）只在训练集OOF预测上计算，避免数据泄漏
  输出 daily_factor_with_scores_strict.parquet
"""

import sys
import os
import argparse
import warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

SELECTED_FACTORS = [
    "dsa_dir",
    "price_vs_dsa_vwap_pct",
    "bbmacd_sign",
    "bbmacd_slope_3",
    "dsa_pivot_pos_01",
    "ret_to_last_low_pct",
    "bars_since_last_high",
    "bbmacd",
    "bbmacd_minus_avg",
    "bbmacd_bandwidth_zscore",
    "vol_zscore_20",
    "vol_stage_cv",
    "vol_zscore_10",
    "vol_ratio_10",
    "days_since_vol_spike",
    "current_stage_amp_pct",
    "prev_stage_amp_pct",
    "current_stage_ret_pct",
    "prev_pivot_code",
    "trend_align_momo",
    "price_vol_coord",
    "momo_vol_coord",
    "low_pos_break_coord",
    "coord_consistency",
    "coord_stage_current",
    "coord_stage_prev",
    "coord_stage_ratio",
]

EXTRA_FEATURES = [
    "weekly_return_score",
    "weekly_risk_score",
    "day_offset",
]

DERIVED_FEATURES = {
    "pivot_pos_x_trend": lambda df: df["dsa_pivot_pos_01"] * df["dsa_dir"],
    "amp_x_pullback": lambda df: df["current_stage_amp_pct"] * df.get("current_pullback_from_stage_extreme_pct", 0),
    "vol_x_stage_amp": lambda df: df["vol_zscore_20"] * df["current_stage_amp_pct"],
}

OPPORTUNITY_TARGET = "ret_5_close_to_close"
RISK_TARGET = "mae_5"
EMBARGO_DAYS = 25

REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "seed": 42,
    "num_leaves": 16,
    "max_depth": 5,
    "min_data_in_leaf": 100,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

CATEGORICAL_FEATURES = []


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: str
    test_end: str


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, func in DERIVED_FEATURES.items():
        try:
            df[name] = func(df)
        except Exception:
            df[name] = np.nan
    return df


def get_all_feature_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in SELECTED_FACTORS:
        if c in df.columns:
            cols.append(c)
    for c in EXTRA_FEATURES:
        if c in df.columns:
            cols.append(c)
    for name in DERIVED_FEATURES:
        if name in df.columns:
            cols.append(name)
    return cols


def build_rolling_splits(df: pd.DataFrame, n_folds: int = 3) -> list:
    trigger_groups = df.groupby("trigger_bar_time")
    trigger_dates = df.groupby("trigger_bar_time")["selection_date"].first()
    sorted_triggers = trigger_dates.sort_values()
    unique_triggers = sorted_triggers.index.tolist()
    n = len(unique_triggers)
    fold_size = n // (n_folds + 1)
    folds = []

    for i in range(n_folds):
        train_end_idx = fold_size * (i + 1)
        test_end_idx = fold_size * (i + 2) if i < n_folds - 1 else n
        if train_end_idx >= n or test_end_idx > n:
            continue

        train_triggers = set(unique_triggers[:train_end_idx])
        test_triggers = set(unique_triggers[train_end_idx:test_end_idx])

        train_end_date = sorted_triggers.iloc[train_end_idx - 1]
        test_start_date = sorted_triggers.iloc[train_end_idx]
        test_end_date = sorted_triggers.iloc[test_end_idx - 1]

        embargo_cutoff = pd.Timestamp(test_start_date) - pd.Timedelta(days=EMBARGO_DAYS)
        train_mask = df["trigger_bar_time"].isin(train_triggers) & (pd.to_datetime(df["selection_date"]) <= embargo_cutoff)
        test_mask = df["trigger_bar_time"].isin(test_triggers)

        train_idx = df.index[train_mask].values
        test_idx = df.index[test_mask].values

        if len(train_idx) < 200 or len(test_idx) < 100:
            continue

        folds.append(Fold(
            name=f"fold_{i + 1}",
            train_idx=train_idx,
            test_idx=test_idx,
            train_end=str(train_end_date)[:10],
            test_end=str(test_end_date)[:10],
        ))
    return folds


def train_lightgbm(df, feature_cols, target_col, fold, params):
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    train_data = lgb.Dataset(
        df.loc[fold.train_idx, feature_cols],
        df.loc[fold.train_idx, target_col],
        categorical_feature=cat_feats,
        free_raw_data=False,
    )
    test_data = lgb.Dataset(
        df.loc[fold.test_idx, feature_cols],
        df.loc[fold.test_idx, target_col],
        reference=train_data,
        categorical_feature=cat_feats,
        free_raw_data=False,
    )
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], callbacks=callbacks)

    y_test = df.loc[fold.test_idx, target_col]
    y_pred = model.predict(df.loc[fold.test_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_test.notna()
    y_test_v = y_test[valid_mask].values
    y_pred_v = y_pred[valid_mask]

    metrics = {
        "fold": fold.name,
        "train_end": fold.train_end,
        "test_end": fold.test_end,
        "n_train": len(fold.train_idx),
        "n_test": len(fold.test_idx),
        "best_iteration": model.best_iteration,
        "test_mae": mean_absolute_error(y_test_v, y_pred_v),
    }

    if len(y_test_v) > 10:
        ic_spearman = stats.spearmanr(y_test_v, y_pred_v)[0]
        metrics["test_ic_spearman"] = ic_spearman
    else:
        metrics["test_ic_spearman"] = np.nan

    return model, metrics


def compute_daily_ic(df, feature_cols, target_col, fold, model) -> pd.DataFrame:
    test_df = df.loc[fold.test_idx].copy()
    test_df["pred"] = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
    valid = test_df[[target_col, "pred", "selection_date"]].dropna()
    if valid.empty:
        return pd.DataFrame()
    rows = []
    for dt, grp in valid.groupby("selection_date"):
        if len(grp) < 10:
            continue
        ic = stats.spearmanr(grp[target_col], grp["pred"])[0]
        rows.append({"date": dt, "ic": ic, "n": len(grp)})
    return pd.DataFrame(rows)


def compute_quintile_report(df, pred_col, label_col) -> pd.DataFrame:
    valid = df[[pred_col, label_col, "selection_date"]].dropna()
    if len(valid) < 100:
        return pd.DataFrame()

    def qcut_daily(g):
        if len(g) < 5:
            return g
        g = g.copy()
        g["quintile"] = pd.qcut(g[pred_col], 5, labels=False, duplicates="drop") + 1
        return g

    valid = valid.groupby("selection_date", group_keys=False).apply(qcut_daily)
    if "quintile" not in valid.columns:
        return pd.DataFrame()

    report = valid.groupby("quintile").agg(
        n=(label_col, "count"),
        mean_label=(label_col, "mean"),
        std_label=(label_col, "std"),
    ).reset_index()
    return report


def train_model_pipeline(df, feature_cols, target_col, model_name, output_dir, params):
    print(f"\n{'=' * 80}")
    print(f"训练 {model_name}（目标: {target_col}）")
    print(f"{'=' * 80}")

    folds = build_rolling_splits(df, n_folds=3)
    print(f"  有效折数: {len(folds)}")
    for f in folds:
        print(f"    {f.name}: train~{f.train_end}, test~{f.test_end}, n_train={len(f.train_idx)}, n_test={len(f.test_idx)}")

    all_daily_ic = []
    all_metrics = []
    all_models = []

    for fold in folds:
        valid_target = df.loc[fold.train_idx, target_col].notna().sum()
        if valid_target < 100:
            print(f"  {fold.name}: 训练集有效标签不足({valid_target})，跳过")
            continue
        model, m = train_lightgbm(df, feature_cols, target_col, fold, params)
        all_metrics.append(m)
        all_models.append(model)
        print(f"  {fold.name}: IC={m['test_ic_spearman']:.4f}, MAE={m['test_mae']:.4f}, iter={m['best_iteration']}")

        daily_ic = compute_daily_ic(df, feature_cols, target_col, fold, model)
        if not daily_ic.empty:
            daily_ic["fold"] = fold.name
            all_daily_ic.append(daily_ic)

    if all_metrics:
        ics = [m["test_ic_spearman"] for m in all_metrics if not np.isnan(m["test_ic_spearman"])]
        if ics:
            ic_mean = np.mean(ics)
            ic_std = np.std(ics)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            print(f"\n  跨折汇总: IC={ic_mean:.4f} ± {ic_std:.4f}, ICIR={icir:.2f}")

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(os.path.join(model_dir, "fold_metrics.csv"), index=False)

    if all_daily_ic:
        pd.concat(all_daily_ic).to_csv(os.path.join(model_dir, "daily_ic.csv"), index=False)

    if all_models:
        all_models[-1].save_model(os.path.join(model_dir, "model.txt"))
        gain = all_models[-1].feature_importance("gain")
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance_gain": gain / gain.sum() if gain.sum() > 0 else gain,
        })
        fi = fi.sort_values("importance_gain", ascending=False).reset_index(drop=True)
        fi.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
        print(f"\n  特征重要性 Top10:")
        for _, row in fi.head(10).iterrows():
            print(f"    {row['feature']:<35} gain={row['importance_gain']:.4f}")

    return all_models, all_metrics, folds


def main():
    parser = argparse.ArgumentParser(description="日线买卖点模型训练")
    parser.add_argument("--sample-limit", type=int, default=0, help="抽样数量（0=全量）")
    parser.add_argument("--n-folds", type=int, default=3, help="滚动折数")
    parser.add_argument("--strict-oos", action="store_true", help="严格OOS模式：只保留最后一折测试集预测")
    parser.add_argument("--dedup", action="store_true", help="去重训练：每个触发只保留day_offset=3的1条记录")
    parser.add_argument("--more-regularization", action="store_true", help="更强正则化：num_leaves=8, min_data=200, feature_fraction=0.5")
    parser.add_argument("--drop-features", type=str, default="", help="要删除的特征列表，逗号分隔")
    parser.add_argument("--minimal-features", action="store_true", help="仅核心+少量辅助特征")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mode_tags = []
    if args.strict_oos:
        mode_tags.append("严格OOS")
    if args.dedup:
        mode_tags.append("去重训练")
    if args.more_regularization:
        mode_tags.append("强正则")
    if args.drop_features:
        mode_tags.append(f"删特征[{args.drop_features}]")
    if args.minimal_features:
        mode_tags.append("最小特征集")
    mode_str = " [" + ", ".join(mode_tags) + "]" if mode_tags else ""

    print("=" * 80)
    print("日线买卖点模型训练" + mode_str)
    print("=" * 80)

    print("\n[1/5] 加载日线因子表...")
    input_path = os.path.join(args.output_dir, "daily_factor_table.parquet")
    df = pd.read_parquet(input_path)
    if args.sample_limit > 0:
        df = df.head(args.sample_limit)

    if args.dedup:
        before = len(df)
        df = df[df["day_offset"] == 3].copy()
        print(f"  去重训练: {before} → {len(df)} 条（每个触发只保留day_offset=3）")

    print(f"  记录数: {len(df)}")
    print(f"  日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")
    print(f"  股票数: {df['ts_code'].nunique()}")
    print(f"  触发点数: {df['trigger_bar_time'].nunique()}")

    print("\n[2/5] 特征工程...")
    df = build_features(df)
    feature_cols = get_all_feature_cols(df)

    if args.drop_features:
        drop_list = [f.strip() for f in args.drop_features.split(",") if f.strip()]
        feature_cols = [c for c in feature_cols if c not in drop_list]
        print(f"  删除特征: {drop_list}")

    if args.minimal_features:
        minimal_set = [
            "price_vs_dsa_vwap_pct", "bbmacd", "bars_since_last_high",
            "weekly_return_score", "day_offset",
        ]
        feature_cols = [c for c in feature_cols if c in minimal_set]

    print(f"  特征数: {len(feature_cols)}")

    params = REG_PARAMS.copy()
    if args.more_regularization:
        params["num_leaves"] = 8
        params["min_data_in_leaf"] = 200
        params["feature_fraction"] = 0.5
        print(f"  强正则化: num_leaves={params['num_leaves']}, min_data={params['min_data_in_leaf']}, feat_frac={params['feature_fraction']}")

    print("\n[3/5] 训练机会模型...")
    ret_models, ret_metrics, folds = train_model_pipeline(
        df, feature_cols, OPPORTUNITY_TARGET, "daily_return_model", args.output_dir, params
    )

    print("\n[4/5] 训练风险模型...")
    risk_models, risk_metrics, _ = train_model_pipeline(
        df, feature_cols, RISK_TARGET, "daily_risk_model", args.output_dir, params
    )

    print("\n[5/5] 生成预测分...")
    if ret_models and risk_models and folds:
        df["daily_opportunity_score"] = np.nan
        df["daily_risk_score"] = np.nan

        if args.strict_oos:
            last_fold = folds[-1]
            if ret_models:
                test_mask = df.index.isin(last_fold.test_idx)
                df.loc[test_mask, "daily_opportunity_score"] = ret_models[-1].predict(
                    df.loc[test_mask, feature_cols], num_iteration=ret_models[-1].best_iteration
                )
            if risk_models:
                test_mask = df.index.isin(last_fold.test_idx)
                df.loc[test_mask, "daily_risk_score"] = risk_models[-1].predict(
                    df.loc[test_mask, feature_cols], num_iteration=risk_models[-1].best_iteration
                )

            train_idx_all = np.concatenate([f.train_idx for f in folds[:-1]])
            train_oof_opp = []
            train_oof_risk = []
            for i, fold in enumerate(folds[:-1]):
                if i < len(ret_models):
                    fold_pred = ret_models[i].predict(
                        df.loc[fold.test_idx, feature_cols], num_iteration=ret_models[i].best_iteration
                    )
                    train_oof_opp.extend(fold_pred)
                if i < len(risk_models):
                    fold_pred = risk_models[i].predict(
                        df.loc[fold.test_idx, feature_cols], num_iteration=risk_models[i].best_iteration
                    )
                    train_oof_risk.extend(fold_pred)

            if train_oof_opp:
                opp_train_q70 = np.quantile(train_oof_opp, 0.7)
                opp_train_q30 = np.quantile(train_oof_opp, 0.3)
            else:
                opp_train_q70 = np.nan
                opp_train_q30 = np.nan
            if train_oof_risk:
                risk_train_q70 = np.quantile(train_oof_risk, 0.7)
                risk_train_q30 = np.quantile(train_oof_risk, 0.3)
            else:
                risk_train_q70 = np.nan
                risk_train_q30 = np.nan

            print(f"\n  严格OOS: 训练集OOF阈值（无泄漏）:")
            print(f"    opp_score q70={opp_train_q70:.4f}, q30={opp_train_q30:.4f}")
            print(f"    risk_score q70={risk_train_q70:.4f}, q30={risk_train_q30:.4f}")

            scored = df[df["daily_opportunity_score"].notna()].copy()
            suffix = ""
            if args.dedup:
                suffix += "_dedup"
            if args.more_regularization:
                suffix += "_strongreg"
            if args.drop_features:
                suffix += f"_drop{len(args.drop_features.split(','))}"
            if args.minimal_features:
                suffix += "_minimal"
            output_path = os.path.join(args.output_dir, f"daily_factor_with_scores_strict{suffix}.parquet")

            threshold_info = {
                "opp_train_q70": opp_train_q70,
                "opp_train_q30": opp_train_q30,
                "risk_train_q70": risk_train_q70,
                "risk_train_q30": risk_train_q30,
            }
            for k, v in threshold_info.items():
                scored[k] = v
        else:
            for i, fold in enumerate(folds):
                if i < len(ret_models):
                    test_mask = df.index.isin(fold.test_idx)
                    df.loc[test_mask, "daily_opportunity_score"] = ret_models[i].predict(
                        df.loc[test_mask, feature_cols], num_iteration=ret_models[i].best_iteration
                    )
                if i < len(risk_models):
                    test_mask = df.index.isin(fold.test_idx)
                    df.loc[test_mask, "daily_risk_score"] = risk_models[i].predict(
                        df.loc[test_mask, feature_cols], num_iteration=risk_models[i].best_iteration
                    )

            scored = df[df["daily_opportunity_score"].notna()].copy()
            output_path = os.path.join(args.output_dir, "daily_factor_with_scores.parquet")

        scored.to_parquet(output_path, index=False)
        print(f"  保存: {output_path}")
        print(f"  有预测分的记录: {len(scored)}")

        if args.strict_oos:
            oos_start = str(scored["selection_date"].min())[:10]
            oos_end = str(scored["selection_date"].max())[:10]
            print(f"  严格OOS时间范围: {oos_start} ~ {oos_end}")
            n_triggers = scored["trigger_bar_time"].nunique()
            print(f"  严格OOS触发点数: {n_triggers}")

        print("\n  预测分统计:")
        for col in ["daily_opportunity_score", "daily_risk_score"]:
            if col in scored.columns:
                s = scored[col]
                print(f"    {col}: mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")

        print("\n  按 day_offset 分组的预测效果:")
        for offset in sorted(scored["day_offset"].unique()):
            sub = scored[scored["day_offset"] == offset]
            if sub.empty:
                continue
            opp_ic = stats.spearmanr(sub["daily_opportunity_score"], sub[OPPORTUNITY_TARGET])[0] if sub[OPPORTUNITY_TARGET].notna().sum() > 10 else np.nan
            risk_ic = stats.spearmanr(sub["daily_risk_score"], sub[RISK_TARGET])[0] if sub[RISK_TARGET].notna().sum() > 10 else np.nan
            avg_opp = sub[OPPORTUNITY_TARGET].mean()
            avg_risk = sub[RISK_TARGET].mean()
            print(f"    T+{offset}: n={len(sub)}, opp_IC={opp_ic:.4f}, risk_IC={risk_ic:.4f}, avg_ret5={avg_opp:.2%}, avg_mae5={avg_risk:.2%}")

        print("\n  机会模型分组报告（按预测分5分组）:")
        opp_report = compute_quintile_report(scored, "daily_opportunity_score", OPPORTUNITY_TARGET)
        if not opp_report.empty:
            print(f"    {'分组':>4} {'样本':>6} {'均值':>8} {'标准差':>8}")
            print(f"    {'-' * 30}")
            for _, row in opp_report.iterrows():
                print(f"    {int(row['quintile']):>4} {int(row['n']):>6} {row['mean_label']:>7.2%} {row['std_label']:>7.2%}")

        print("\n  风险模型分组报告（按预测分5分组）:")
        risk_report = compute_quintile_report(scored, "daily_risk_score", RISK_TARGET)
        if not risk_report.empty:
            print(f"    {'分组':>4} {'样本':>6} {'均值':>8} {'标准差':>8}")
            print(f"    {'-' * 30}")
            for _, row in risk_report.iterrows():
                print(f"    {int(row['quintile']):>4} {int(row['n']):>6} {row['mean_label']:>7.2%} {row['std_label']:>7.2%}")
    else:
        print("  模型不足，跳过预测分生成")

    print("\n" + "=" * 80)
    print("日线买卖点模型训练完成" + (" [严格OOS模式]" if args.strict_oos else ""))
    print("=" * 80)


if __name__ == "__main__":
    main()
