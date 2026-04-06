"""GBDT因子重要性分析（排除未来信息，时序CV，SHAP，消融实验）

场景:
  A: T0入场回归 (目标=hold_ret, 特征=T0因子)
  B: T+1入场回归 (目标=hold_ret_t1, 特征=T0+T1因子)
  C: 分类版本 (目标=win_label, 特征同A/B)

输出:
  gbdt_feature_importance.csv   L1+L2+L3综合排名
  gbdt_shap_summary.csv        SHAP全局摘要
  gbdt_group_return.csv        五分组收益(验证单调性)
  gbdt_ic_by_fold.csv          各折IC/RankIC
  gbdt_ablation.csv            消融实验
  gbdt_shap_beeswarm.png       SHAP蜂群图(每场景)
"""
import os, sys, warnings, time, argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

import lightgbm as lgb
import shap

DATA_FILE = 'breakout_events_full.csv'
OUT_DIR = 'gbdt_analysis'

os.makedirs(OUT_DIR, exist_ok=True)

FUTURE_FACTOR_SET = {
    'bars_to_first_dir_flip', 'max_ret_before_flip', 'max_dd_before_flip',
    'flip1_close', 'flip1_rope_dir', 'flip1_dist_to_rope_atr',
    'flip1_position_in_structure', 'flip1_state_score_total',
    'first_dir_flip_time', 'first_dir_flip_type',
    'ret_5', 'ret_10', 'ret_20',
}

TIME_PROXY_SET = {
    'event_index', 'event_seq',
}

SCALE_PROXY_SET = {
    'open', 'high', 'low', 'close', 'amount', 'vol',
    'nearest_high', 'nearest_low', 'rope', 'upper', 'lower', 'atr_raw',
}

BASE_EXCLUDE = {
    'event_id', 'symbol', 'freq', 'event_time', 't1_time',
    'breakout_quality_grade',
} | FUTURE_FACTOR_SET | TIME_PROXY_SET | SCALE_PROXY_SET


def load_and_prepare(nrows=None):
    print(f"加载数据: {DATA_FILE}" + (f" (前{nrows}条)" if nrows else ""))
    df = pd.read_csv(DATA_FILE, nrows=nrows)
    print(f"  原始样本: {len(df)}")

    df['hold_ret'] = (df['flip1_close'] - df['close']) / df['close']
    df['hold_ret_t1'] = (df['flip1_close'] - df['t1_close']) / df['t1_close']
    df['ret_t1'] = (df['t1_close'] - df['close']) / df['close']
    df['dir_consistent'] = (df['t1_rope_dir'] == df['rope_dir']).astype(int)
    df['win_label'] = (df['hold_ret'] > 0).astype(int)

    if 'event_time' in df.columns:
        df['_time'] = pd.to_datetime(df['event_time'])
    elif 'bar_time' in df.columns:
        df['_time'] = pd.to_datetime(df['bar_time'])
    else:
        raise ValueError("找不到时间列")

    return df.sort_values('_time').reset_index(drop=True)


def get_t0_factors(df):
    targets = {'ret_5', 'ret_10', 'ret_20', 'hold_ret', 'hold_ret_t1',
               'ret_t1', 'win_label'}
    factors = [c for c in df.columns
               if c not in BASE_EXCLUDE and c not in targets
               and not c.startswith('t1_') and not c.startswith('_')
               and df[c].dtype in ['float64', 'int64']]
    return sorted(factors)


def get_t1_factors(df):
    t0 = get_t0_factors(df)
    extra = [c for c in ['dir_consistent', 'ret_t1'] if c in df.columns and c not in t0]
    return t0 + extra


def winsorize_series(s, limits=(0.01, 0.99)):
    if isinstance(s, pd.Series):
        lo, hi = s.quantile(list(limits))
        return s.clip(lo, hi)
    arr = np.asarray(s, dtype=float)
    lo = np.nanpercentile(arr, limits[0] * 100)
    hi = np.nanpercentile(arr, limits[1] * 100)
    return np.clip(arr, lo, hi)


def prepare_features(df, feature_cols, target_col, task='regression'):
    sub = df[feature_cols + [target_col, '_time']].dropna(subset=[target_col])
    if len(sub) < 500:
        return None, None, None, None

    X = sub[feature_cols].copy()
    y = sub[target_col].copy()
    times = sub['_time'].values

    high_missing = X.isnull().mean() > 0.30
    if high_missing.any():
        dropped = X.columns[high_missing].tolist()
        print(f"  剔除高缺失列({dropped})")
        X = X.loc[:, ~high_missing]
        feature_cols = [c for c in feature_cols if c in X.columns]

    medians = X.median()
    X = X.fillna(medians)

    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = winsorize_series(X[col])

    feature_cols = list(X.columns)
    return X.values, y.values, times, feature_cols


class TimeSeriesCV:
    def __init__(self, times, n_folds=10, train_years=2, test_months=6):
        self.times = times
        self.n_folds = n_folds
        self.train_days = train_years * 365
        self.test_days = test_months * 30

    def split(self):
        n = len(self.times)
        folds = []
        step = (n - self.train_days) // self.n_folds
        for i in range(self.n_folds):
            test_start = self.train_days + i * step
            test_end = min(test_start + self.test_days, n)
            if test_end >= n:
                break
            train_idx = np.arange(test_start)
            test_idx = np.arange(test_start, test_end)
            folds.append((train_idx, test_idx))
        return folds


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       feature_names, params, task='regression'):
    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=feature_names, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test,
                        feature_name=feature_names, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_estimators', 500),
        valid_sets=[dtest],
        callbacks=[
            lgb.log_evaluation(0),
            lgb.early_stopping(params.get('early_stopping', 50), verbose=False),
        ],
    )

    pred = model.predict(X_test)

    ic = np.corrcoef(pred, y_test)[0, 1] if np.std(pred) > 0 else 0
    rank_ic, _ = spearmanr(pred, y_test) if np.std(pred) > 0 else (0, 1)

    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))

    if task == 'classification':
        acc = np.mean((pred > 0.5).astype(int) == y_test)
        metrics = dict(ic=ic, rank_ic=rank_ic, mae=mae, rmse=rmse, acc=acc)
    else:
        metrics = dict(ic=ic, rank_ic=rank_ic, mae=mae, rmse=rmse)

    return model, pred, metrics


def run_single_scene(df, scene_name, feature_cols, target_col, task='regression',
                     n_folds=10, small=False):
    print(f"\n{'#'}" * 70)
    print(f"# 场景 {scene_name}: target={target_col}, task={task}")
    print(f"# 特征数: {len(feature_cols)}")
    print(f"{'#'}" * 70)

    X, y, times, feat_used = prepare_features(df, feature_cols, target_col, task)
    if X is None:
        print(f"  有效样本不足")
        return None

    print(f"  有效样本: {len(y)}, 时间范围: {times[0]} ~ {times[-1]}")

    if task == 'regression':
        params = {
            'objective': 'regression',
            'metric': ['mae', 'rmse'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 800,
            'min_child_samples': max(100, int(len(y) * 0.001)),
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'early_stopping': 50,
        }
    else:
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 800,
            'min_child_samples': max(100, int(len(y) * 0.001)),
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'early_stopping': 50,
        }

    actual_n_folds = min(n_folds, 5) if small else n_folds
    cv = TimeSeriesCV(times, n_folds=actual_n_folds)
    splits = cv.split()
    print(f"  CV折数: {len(splits)}")

    fold_results = []
    all_preds = np.zeros(len(y))
    all_importance_gain = np.zeros(len(feat_used))
    all_importance_split = np.zeros(len(feat_used))

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        Xt, yt = X[train_idx], y[train_idx]
        Xv, yv = X[test_idx], y[test_idx]

        model, pred, metrics = train_and_evaluate(
            Xt, yt, Xv, yv, feat_used, params, task
        )

        all_preds[test_idx] = pred
        all_importance_gain += model.feature_importance(importance_type='gain')
        all_importance_split += model.feature_importance(importance_type='split')

        fold_results.append({
            'fold': fold_i + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            **metrics,
        })

        print(f"  Fold{fold_i + 1}: IC={metrics['ic']:.4f}  RankIC={metrics['rank_ic']:.4f}  "
              f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}"
              + (f"  Acc={metrics['acc']:.4f}" if task == 'classification' else ''))

    avg_gain = all_importance_gain / len(splits)
    avg_split = all_importance_split / len(splits)

    overall_ic = np.corrcoef(all_preds, y)[0, 1]
    overall_rank_ic, _ = spearmanr(all_preds, y)
    print(f"\n  === 整体: IC={overall_ic:.4f}  RankIC={overall_rank_ic:.4f} ===")

    imp_df = pd.DataFrame({
        'feature': feat_used,
        'gain': avg_gain,
        'split': avg_split,
    })
    imp_df['gain_pct'] = imp_df['gain'] / imp_df['gain'].sum() * 100
    imp_df['split_pct'] = imp_df['split'] / imp_df['split'].sum() * 100
    imp_df = imp_df.sort_values('gain', ascending=False).reset_index(drop=True)

    print(f"\n  --- Top15 Feature Importance (Gain) ---")
    for _, r in imp_df.head(15).iterrows():
        print(f"    {r['feature']:<40}  gain={r['gain']:>10.1f} ({r['gain_pct']:>5.1f}%)  "
              f"split={r['split']:>8.0f}")

    group_res = compute_group_returns(all_preds, y, scene_name)

    shap_res = compute_shap(X, y, all_preds, feat_used, model, scene_name, small)

    abl_res = run_ablation(X, y, times, feat_used, params, task, scene_name, small)

    fold_df = pd.DataFrame(fold_results)
    out_path = f'{OUT_DIR}/ic_{scene_name}.csv'
    fold_df.to_csv(out_path, index=False)
    print(f"\n  已保存IC: {out_path}")

    return dict(
        scene=scene_name,
        target=target_col,
        task=task,
        n_samples=len(y),
        n_features=len(feat_used),
        overall_ic=round(overall_ic, 4),
        overall_rank_ic=round(overall_rank_ic, 4),
        importance=imp_df,
        shap=shap_res,
        group=group_res,
        ablation=abl_res,
        folds=fold_df,
    )


def compute_group_returns(preds, actuals, label):
    valid = ~(np.isnan(preds) | np.isnan(actuals))
    p, a = preds[valid], actuals[valid]
    if len(p) < 200:
        return None

    try:
        cuts = pd.qcut(p, q=5, duplicates='drop')
        n_grp = len(cuts.cat.categories)
        labels = [f'Q{i + 1}' for i in range(n_grp)]
        grp = pd.qcut(p, q=n_grp, labels=labels, duplicates='drop')
    except Exception:
        return None

    rows = []
    for g in sorted(grp.cat.categories):
        mask = grp == g
        sg = a[mask]
        rows.append({
            'scene': label,
            'group': str(g),
            'n': mask.sum(),
            'mean_pred': round(p[mask].mean(), 4),
            'mean_actual': round(sg.mean(), 4),
            'median_actual': round(np.median(sg), 4),
            'std_actual': round(sg.std(), 4),
            'win_rate': round((sg > 0).mean(), 4),
        })
    df = pd.DataFrame(rows)
    means = df['mean_actual'].values
    df['monotonic'] = (
        all(means[i] <= means[i + 1] for i in range(len(means) - 1)) or
        all(means[i] >= means[i + 1] for i in range(len(means) - 1))
    )
    df['spread'] = df['mean_actual'].max() - df['mean_actual'].min()

    out_path = f'{OUT_DIR}/group_{label}.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  分组收益:")
    for _, r in df.iterrows():
        mono = "Y" if r['monotonic'] else "N"
        print(f"    {r['group']}  预测均值={r['mean_pred']:+.4f}  "
              f"实际均值={r['mean_actual']:+.4f}  胜率={r['win_rate']:.2%}  N={r['n']}")
    return df


def compute_shap(X, y, preds, feat_names, ref_model, label, small=False):
    print(f"\n  计算SHAP值...")
    sample_size = min(2000, len(X)) if small else min(10000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(ref_model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_val = shap_values.mean(axis=0)

    summary = pd.DataFrame({
        'feature': feat_names,
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_val,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    out_path = f'{OUT_DIR}/shap_{label}.csv'
    summary.to_csv(out_path, index=False)

    print(f"  --- Top15 SHAP (mean|SHAP|) ---")
    for _, r in summary.head(15).iterrows():
        direction = "+" if r['mean_shap'] > 0 else "-"
        print(f"    {r['feature']:<40}  |SHAP|={r['mean_abs_shap']:.4f}  "
              f"方向={direction}{abs(r['mean_shap']):.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, max(6, len(feat_names) * 0.3)))
        shap.summary_plot(shap_values, X_sample, feature_names=feat_names,
                          show=False, max_display=20)
        plt.tight_layout()
        png_path = f'{OUT_DIR}/shap_beeswarm_{label}.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  SHAP图已保存: {png_path}")
    except Exception as e:
        print(f"  SHAP绘图失败: {e}")

    return summary


def run_ablation(X, y, times, feat_names, params, task, label, small):
    print(f"\n  运行消融实验 ({len(feat_names)} 个因子)...")
    actual_n_folds = 5 if small else 8
    cv = TimeSeriesCV(times, n_folds=actual_n_folds)
    splits = cv.split()

    baseline_metrics = []
    for train_idx, test_idx in splits:
        _, _, m = train_and_evaluate(
            X[train_idx], y[train_idx],
            X[test_idx], y[test_idx],
            feat_names, params, task
        )
        baseline_metrics.append(m)

    base_ic = np.mean([m['ic'] for m in baseline_metrics])
    base_rank_ic = np.mean([m['rank_ic'] for m in baseline_metrics])

    results = []
    total = len(feat_names)
    for i, col in enumerate(feat_names):
        keep_idx = [j for j, fn in enumerate(feat_names) if fn != col]
        if len(keep_idx) < 3:
            continue
        X_reduced = X[:, keep_idx]
        reduced_names = [feat_names[j] for j in keep_idx]

        fold_ics = []
        for train_idx, test_idx in splits:
            try:
                _, _, m = train_and_evaluate(
                    X_reduced[train_idx], y[train_idx],
                    X_reduced[test_idx], y[test_idx],
                    reduced_names, params, task
                )
                fold_ics.append(m['ic'])
            except Exception:
                fold_ics.append(0.0)

        drop_ic = base_ic - np.mean(fold_ics)
        results.append({
            'removed_feature': col,
            'baseline_ic': round(base_ic, 4),
            'ablated_ic': round(np.mean(fold_ics), 4),
            'ic_drop': round(drop_ic, 4),
        })
        if (i + 1) % 5 == 0 or i == total - 1:
            print(f"    进度: {i + 1}/{total}")

    abl_df = pd.DataFrame(results).sort_values('ic_drop', ascending=False)
    out_path = f'{OUT_DIR}/ablation_{label}.csv'
    abl_df.to_csv(out_path, index=False)

    print(f"  --- 消融Top15 (移除后IC下降最多) ---")
    for _, r in abl_df.head(15).iterrows():
        print(f"    移除 {r['removed_feature']:<35}  IC变化: {r['baseline_ic']:+.4f} -> "
              f"{r['ablated_ic']:+.4f}  (Δ={r['ic_drop']:+.4f})")
    return abl_df


def main():
    parser = argparse.ArgumentParser(description='GBDT因子重要性分析')
    parser.add_argument('--small', action='store_true', help='小批量验证(5000条)')
    parser.add_argument('--folds', type=int, default=10, help='CV折数')
    args = parser.parse_args()

    nrows = 5000 if args.small else None
    df = load_and_prepare(nrows)

    t0_feats = get_t0_factors(df)
    t1_feats = get_t1_factors(df)
    print(f"\nT0因子数: {len(t0_feats)}")
    print(f"T+1因子数: {len(t1_feats)}")
    print(f"  排除-未来信息({len(FUTURE_FACTOR_SET)}): {sorted(FUTURE_FACTOR_SET)}")
    print(f"  排除-时间代理({len(TIME_PROXY_SET)}): {sorted(TIME_PROXY_SET)}")
    print(f"  排除-规模代理({len(SCALE_PROXY_SET)}): {sorted(SCALE_PROXY_SET)}")
    print(f"  T0可用: {t0_feats}")

    all_results = []

    res_a = run_single_scene(
        df, 'sceneA_T0_reg', t0_feats, 'hold_ret',
        task='regression', n_folds=args.folds, small=args.small
    )
    if res_a:
        all_results.append(res_a)

    res_b = run_single_scene(
        df, 'sceneB_T1_reg', t1_feats, 'hold_ret_t1',
        task='regression', n_folds=args.folds, small=args.small
    )
    if res_b:
        all_results.append(res_b)

    res_c = run_single_scene(
        df, 'sceneC_T0_cls', t0_feats, 'win_label',
        task='classification', n_folds=args.folds, small=args.small
    )
    if res_c:
        all_results.append(res_c)

    res_d = run_single_scene(
        df, 'sceneD_T1_cls', t1_feats, 'win_label',
        task='classification', n_folds=args.folds, small=args.small
    )
    if res_d:
        all_results.append(res_d)

    if all_results:
        summary_rows = []
        for r in all_results:
            summary_rows.append({
                'scene': r['scene'],
                'task': r['task'],
                'target': r['target'],
                'n_samples': r['n_samples'],
                'n_features': r['n_features'],
                'overall_ic': r['overall_ic'],
                'overall_rank_ic': r['overall_rank_ic'],
            })
        summary_df = pd.DataFrame(summary_rows)
        sum_path = f'{OUT_DIR}/summary.csv'
        summary_df.to_csv(sum_path, index=False)

        all_imp = []
        for r in all_results:
            imp = r['importance'][['feature', 'gain', 'gain_pct', 'split', 'split_pct']].copy()
            imp['scene'] = r['scene']
            all_imp.append(imp)
        combined_imp = pd.concat(all_imp, ignore_index=True)
        imp_path = f'{OUT_DIR}/gbdt_feature_importance.csv'
        combined_imp.to_csv(imp_path, index=False)

        print(f"\n{'#' * 70}")
        print("# 总览")
        print(f"{'#' * 70}")
        print(summary_df.to_string(index=False))
        print(f"\n结果保存在: {OUT_DIR}/")


if __name__ == '__main__':
    main()
