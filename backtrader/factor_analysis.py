"""因子与收益关系探索分析（修正版：排除未来信息，按入场时点分场景）"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings, os, argparse

warnings.filterwarnings('ignore')

DATA_FILE = 'breakout_events_full.csv'
OUT_DIR = 'factor_analysis'

os.makedirs(OUT_DIR, exist_ok=True)

FUTURE_FACTOR_SET = {
    'bars_to_first_dir_flip', 'max_ret_before_flip', 'max_dd_before_flip',
    'flip1_close', 'flip1_rope_dir', 'flip1_dist_to_rope_atr',
    'flip1_position_in_structure', 'flip1_state_score_total',
    'first_dir_flip_time', 'first_dir_flip_type',
}

T0_BASE_EXCLUDE = {
    'event_id', 'symbol', 'freq', 'event_time', 't1_time',
    'first_dir_flip_time', 'first_dir_flip_type', 'breakout_quality_grade',
} | FUTURE_FACTOR_SET

T1_CONFIRM_COLS = {'t1_open', 't1_high', 't1_low', 't1_close', 't1_rope_dir'}


def load_data(nrows=None):
    print(f"加载数据: {DATA_FILE}" + (f" (前{nrows}条)" if nrows else ""))
    df = pd.read_csv(DATA_FILE, nrows=nrows)
    print(f"  总样本: {len(df)}")
    return df


def compute_targets(df):
    """计算各场景的目标变量（不含未来信息）"""
    # 场景A：T0收盘入场 → Tf翻转退出
    df['hold_ret'] = (df['flip1_close'] - df['close']) / df['close']
    # 场景B：T+1收盘确认入场 → Tf翻转退出
    df['hold_ret_t1'] = (df['flip1_close'] - df['t1_close']) / df['t1_close']
    # T+1当日收益（用于构建确认信号）
    df['ret_t1'] = (df['t1_close'] - df['close']) / df['close']
    # 方向一致性
    df['dir_consistent'] = (df['t1_rope_dir'] == df['rope_dir']).astype(int)
    print(f"  目标变量已计算: hold_ret, hold_ret_t1, ret_t1, dir_consistent")
    return df


def get_t0_factors(df):
    """T0时刻已知因子（排除所有未来信息）"""
    base_targets = {'ret_5', 'ret_10', 'ret_20', 'hold_ret', 'hold_ret_t1', 'ret_t1'}
    factors = [c for c in df.columns
               if c not in T0_BASE_EXCLUDE and c not in base_targets
               and not c.startswith('t1_')
               and df[c].dtype in ['float64', 'int64']]
    return sorted(factors)


def get_t1_confirm_factors(df):
    """T+1确认因子 = T0因子 + T1信号"""
    t0 = get_t0_factors(df)
    extra = ['dir_consistent', 'ret_t1']
    available = [c for c in extra if c in df.columns]
    return t0 + available


def run_correlation(df, factors, targets, label):
    """单因子相关性分析"""
    print(f"\n{'='*65}")
    print(f"【{label}】单因子相关性分析")
    print(f"因子数: {len(factors)}, 目标变量: {targets}")

    results = []
    for factor in factors:
        if factor not in df.columns:
            continue
        for target in targets:
            if target not in df.columns:
                continue
            pair = df[[factor, target]].dropna()
            if len(pair) < 100:
                continue
            pr, pv = stats.pearsonr(pair[factor], pair[target])
            sr, sp = stats.spearmanr(pair[factor], pair[target])
            results.append({
                'scene': label,
                'factor': factor,
                'target': target,
                'n': len(pair),
                'pearson_r': round(pr, 4),
                'pearson_p': pv,
                'spearman_r': round(sr, 4),
            })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("  无有效结果")
        return res_df

    res_df['abs_pearson'] = res_df['pearson_r'].abs()
    res_df = res_df.sort_values(['abs_pearson', 'factor'], ascending=[False, True])

    top = res_df.drop_duplicates(subset='factor').head(20)
    print(f"\n--- Top20 因子（按|Pearson|排序）---")
    print(f"{'因子':<35} {'目标':>12} {'Pearson':>8} {'Spearman':>9} {'样本':>7}")
    print("-" * 75)
    for _, r in top.iterrows():
        sig = "***" if r['pearson_p'] < 0.001 else ("**" if r['pearson_p'] < 0.01 else ("*" if r['pearson_p'] < 0.05 else ""))
        print(f"{r['factor']:<35} {r['target']:>12} {r['pearson_r']:>8.4f}{sig} {r['spearman_r']:>9.4f} {r['n']:>7}")

    out_path = f'{OUT_DIR}/corr_{label}.csv'
    res_df.to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")
    return res_df


def run_group_analysis(df, factors, targets, label, top_n=25):
    """五分位数分组收益分析"""
    print(f"\n{'='*65}")
    print(f"【{label}】五分位数分组收益分析")

    all_groups = []
    for factor in factors[:top_n]:
        if factor not in df.columns:
            continue
        for target in targets:
            if target not in df.columns:
                continue
            valid = df[[factor, target]].dropna()
            if len(valid) < 200:
                continue

            valid = valid.copy()
            try:
                cuts = pd.qcut(valid[factor], q=5, duplicates='drop')
                n_groups = len(cuts.cat.categories)
                if n_groups < 3:
                    continue
                labels = [f'Q{i+1}' for i in range(n_groups)]
                valid['group'] = pd.qcut(valid[factor], q=n_groups, labels=labels, duplicates='drop')
            except Exception:
                continue

            grp = valid.groupby('group')[target].agg(['mean', 'median', 'std', 'count'])
            grp['win_rate'] = valid.groupby('group')[target].apply(lambda x: (x > 0).mean())
            grp = grp.reset_index()
            grp['factor'] = factor
            grp['target'] = target

            means = grp['mean'].values
            monotonic = (all(means[i] <= means[i + 1] for i in range(len(means) - 1)) or
                        all(means[i] >= means[i + 1] for i in range(len(means) - 1)))
            grp['monotonic'] = monotonic
            grp['spread'] = grp['mean'].max() - grp['mean'].min()
            all_groups.append(
                grp[['factor', 'target', 'group', 'mean', 'median', 'std', 'count', 'win_rate', 'monotonic', 'spread']]
            )

    if not all_groups:
        print("  无有效分组结果")
        return None

    group_df = pd.concat(all_groups, ignore_index=True)

    summary = []
    for factor in factors[:top_n]:
        fg = group_df[group_df['factor'] == factor]
        if fg.empty:
            continue
        best = fg.loc[fg['spread'].idxmax()]
        q1_row = fg[fg['group'] == 'Q1']
        q5_row = fg[fg['group'] == f'Q{len(fg["group"].unique())}']
        summary.append({
            'factor': factor,
            'best_target': best['target'],
            'spread': round(best['spread'], 4),
            'monotonic': best['monotonic'],
            'q1_mean': round(q1_row['mean'].values[0], 4) if len(q1_row) else None,
            'q5_mean': round(q5_row['mean'].values[0], 4) if len(q5_row) else None,
        })

    sum_df = pd.DataFrame(summary).sort_values('spread', ascending=False)

    print(f"\n--- 分组收益 spread Top15 ---")
    print(f"{'因子':<35} {'目标':>12} {'Spread':>8} {'单调':>5} {'Q1均值':>9} {'Q5均值':>9}")
    print("-" * 80)
    for _, r in sum_df.head(15).iterrows():
        mono = "Y" if r['monotonic'] else "N"
        print(f"{r['factor']:<35} {r['best_target']:>12} {r['spread']:>8.4f} {mono:>5} {r['q1_mean']:>9.4f} {r['q5_mean']:>9.4f}")

    out_path = f'{OUT_DIR}/group_{label}.csv'
    group_df.to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")
    return group_df


def run_interaction(df, label):
    """交互效应分析"""
    print(f"\n{'='*65}")
    print(f"【{label}】交互效应分析")

    results = []

    if label == 'sceneA':
        targets = ['hold_ret', 'ret_5', 'ret_10', 'ret_20']
    elif label == 'sceneB':
        targets = ['hold_ret_t1', 'hold_ret', 'ret_5', 'ret_10', 'ret_20']
    else:
        targets = ['ret_5', 'ret_10', 'ret_20']

    # 质量等级 × 收益
    print("--- 质量等级 ---")
    for grade in ['B', 'C', 'D']:
        sub = df[df['breakout_quality_grade'] == grade]
        for t in targets:
            valid = sub[t].dropna()
            if len(valid) > 50:
                results.append({
                    'analysis': 'quality_grade', 'group': f'grade_{grade}',
                    'target': t, 'n': len(valid),
                    'mean_ret': round(valid.mean(), 4),
                    'median_ret': round(valid.median(), 4),
                    'win_rate': round((valid > 0).mean(), 4),
                    'std': round(valid.std(), 4),
                })

    # rope_dir × 收益
    print("--- 突破方向(rope_dir) ---")
    for rd_val in sorted(df['rope_dir'].dropna().unique()):
        sub = df[df['rope_dir'] == rd_val]
        dlabel = "多头" if rd_val == 1.0 else ("空头" if rd_val == -1.0 else f"other_{rd_val}")
        for t in targets:
            valid = sub[t].dropna()
            if len(valid) > 50:
                results.append({
                    'analysis': 'rope_dir', 'group': dlabel,
                    'target': t, 'n': len(valid),
                    'mean_ret': round(valid.mean(), 4),
                    'median_ret': round(valid.median(), 4),
                    'win_rate': round((valid > 0).mean(), 4),
                    'std': round(valid.std(), 4),
                })

    # high_break_flag × 收益
    print("--- 突破类型(high_break_flag) ---")
    for hbf in sorted(df['high_break_flag'].dropna().unique()):
        sub = df[df['high_break_flag'] == hbf]
        hlabel = "高点突破" if hbf == 1 else "非高点突破"
        for t in targets:
            valid = sub[t].dropna()
            if len(valid) > 50:
                results.append({
                    'analysis': 'high_break_flag', 'group': hlabel,
                    'target': t, 'n': len(valid),
                    'mean_ret': round(valid.mean(), 4),
                    'median_ret': round(valid.median(), 4),
                    'win_rate': round((valid > 0).mean(), 4),
                    'std': round(valid.std(), 4),
                })

    # watch_flag × 收益
    print("--- 观察标记(watch_flag) ---")
    for wf in sorted(df['breakout_watch_flag'].dropna().unique()):
        sub = df[df['breakout_watch_flag'] == wf]
        wlabel = "观察(B级)" if wf == 1 else "普通"
        for t in targets:
            valid = sub[t].dropna()
            if len(valid) > 50:
                results.append({
                    'analysis': 'watch_flag', 'group': wlabel,
                    'target': t, 'n': len(valid),
                    'mean_ret': round(valid.mean(), 4),
                    'median_ret': round(valid.median(), 4),
                    'win_rate': round((valid > 0).mean(), 4),
                    'std': round(valid.std(), 4),
                })

    # 场景B额外：T+1方向一致性
    if label == 'sceneB':
        print("--- T+1方向一致性(仅场景B) ---")
        for cons in [0, 1]:
            clabel = "方向一致" if cons == 1 else "方向翻转"
            sub = df[df['dir_consistent'] == cons]
            for t in targets:
                valid = sub[t].dropna()
                if len(valid) > 50:
                    results.append({
                        'analysis': 't1_consistency', 'group': clabel,
                        'target': t, 'n': len(valid),
                        'mean_ret': round(valid.mean(), 4),
                        'median_ret': round(valid.median(), 4),
                        'win_rate': round((valid > 0).mean(), 4),
                        'std': round(valid.std(), 4),
                    })

    int_df = pd.DataFrame(results)
    out_path = f'{OUT_DIR}/interaction_{label}.csv'
    int_df.to_csv(out_path, index=False)

    print(f"\n--- 汇总表 ---")
    print(f"{'分析维度':<18} {'分组':<14} {'目标':>12} {'样本':>7} {'均值收益':>10} {'胜率':>7} {'标准差':>8}")
    print("-" * 82)
    for _, r in int_df.iterrows():
        print(f"{r['analysis']:<18} {r['group']:<14} {r['target']:>12} {r['n']:>7} {r['mean_ret']:>10.4f} {r['win_rate']:>7.2%} {r['std']:>8.4f}")

    print(f"\n已保存: {out_path}")
    return int_df


def main():
    parser = argparse.ArgumentParser(description='因子-收益关系探索（修正版：无未来信息）')
    parser.add_argument('--small', action='store_true', help='小批量验证(5000条)')
    args = parser.parse_args()

    nrows = 5000 if args.small else None
    df = load_data(nrows)
    df = compute_targets(df)

    t0_factors = get_t0_factors(df)
    t1_factors = get_t1_confirm_factors(df)
    print(f"\nT0已知因子数: {len(t0_factors)}")
    print(f"T+1确认因子数: {len(t1_factors)}")
    print(f"被排除的未来信息因子: {sorted(FUTURE_FACTOR_SET)}")

    # ===== 场景A：T0入场，仅用T0因子，目标=hold_ret =====
    corr_a = run_correlation(df, t0_factors, ['hold_ret', 'ret_5', 'ret_10', 'ret_20'], 'sceneA_T0_entry')
    run_group_analysis(df, t0_factors, ['hold_ret', 'ret_5', 'ret_10', 'ret_20'], 'sceneA_T0_entry')
    run_interaction(df, 'sceneA')

    # ===== 场景B：T+1入场，用T0+T1因子，目标=hold_ret_t1 =====
    corr_b = run_correlation(df, t1_factors, ['hold_ret_t1', 'hold_ret', 'ret_5', 'ret_10', 'ret_20'], 'sceneB_T1_entry')
    run_group_analysis(df, t1_factors, ['hold_ret_t1', 'hold_ret', 'ret_5', 'ret_10', 'ret_20'], 'sceneB_T1_entry')
    run_interaction(df, 'sceneB')

    # ===== 基线：T0因子 vs 固定窗口收益（对照）=====
    run_correlation(df, t0_factors, ['ret_5', 'ret_10', 'ret_20'], 'baseline_fixed_window')

    print(f"\n{'='*65}")
    print("全部完成！结果保存在 {OUT_DIR}/ 目录")


if __name__ == '__main__':
    main()
