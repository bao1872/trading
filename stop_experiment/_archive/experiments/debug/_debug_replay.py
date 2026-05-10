#!/usr/bin/env python3
"""
[调试工具] 逐日追踪 replay vs backtest 对账差异

Purpose:
    逐日对比 replay 模块 (06_daily_inference_replay.py) 和 backtest 引擎 (dynamic_exit_backtest_v2.py)
    的决策输出，定位 holdings_before / sells / buys 不一致的根因。

Pipeline Position:
    调试工具（开发/排查时使用）。
    上游: dynamic_exit_backtest_v2.py, 06_daily_inference_replay.py
    下游: —

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - Console: 逐日对账结果（HB/S/B 是否一致）

How to Run:
    python stop_experiment/experiments/debug/_debug_replay.py

Side Effects:
    - 只读，无文件输出
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd, numpy as np
from stop_experiment.pipeline.stop_config import OUTPUT_DIR, V1_PARAMS
from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data, run_backtest
from stop_experiment.backtest.simple_backtest import score_stocks

# import from the replay module using importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    'daily_inference_replay',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline', '06_daily_inference_replay.py'))
replay_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_mod)
build_daily_candidate_snapshot = replay_mod.build_daily_candidate_snapshot
decide_daily_actions = replay_mod.decide_daily_actions

test_df, price, td, prev_close, pred_lookup = _load_data(candidate_obs_days=V1_PARAMS['candidate_obs_days'])
result = run_backtest(test_df, price, td, prev_close, pred_lookup, max_stocks=10, strategy='sell_score',
                      exit_mode='model_exit', stop_loss=V1_PARAMS['stop_loss'],
                      buy_cls_exit_threshold=V1_PARAMS['buy_cls_exit_threshold'],
                      debug_snapshots=True, strict=True)
smap = {s['date']: s for s in result['snapshots']}

df_all = pd.read_parquet(os.path.join(OUTPUT_DIR, 'full_test_predictions.parquet'))
df_all['obs_date'] = pd.to_datetime(df_all['obs_date'])
df_all = df_all[df_all['obs_day'].isin([1,2,3])].copy()
df_all = score_stocks(df_all, 'sell_score')
score_col = 'score'

pred_indexed = {}
for (sid, dt), pred in pred_lookup.items():
    pred_indexed[(int(sid), dt)] = pred

trading_dates_list = sorted(smap.keys())

holdings = {}
print('日期       | holdings_before       | sells                | buys                 | match')
print('-'*100)

for tdate in sorted(trading_dates_list):
    snap = smap.get(tdate)
    if snap is None:
        continue
    if tdate > pd.Timestamp('2026-03-20'):
        break

    prev_idx = None
    for i, td_date in enumerate(trading_dates_list):
        if td_date == tdate:
            prev_idx = i - 1 if i > 0 else None
            break
    prev_date = trading_dates_list[prev_idx] if prev_idx is not None else None

    candidates = build_daily_candidate_snapshot(tdate, df_all, score_col)

    for code, h in holdings.items():
        h['cur_ret'] = None
        if tdate in price.index:
            close_s = price.loc[tdate, 'close']
            if code in close_s.index:
                close_p = close_s[code]
                if pd.notna(close_p) and h.get('buy_price') and h['buy_price'] > 0:
                    h['cur_ret'] = (close_p - h['buy_price']) / h['buy_price']

    new_holdings, buys, sells, sell_reasons = decide_daily_actions(
        tdate, dict(holdings), candidates, pred_indexed, prev_date,
        price_pivot=price, trading_dates=td,
    )

    bt_hb = snap.get('holdings_before', {})  # 已含前日 pending_orders 执行
    bt_sell_reasons = snap.get('sell_reasons', {})
    bt_buys = snap.get('buys', [])  # 当日新 pending_orders (=buys)

    # RP holdings_before = holdings (before decide)- matches BT holdings_before
    rp_hb_codes = set(v.get('ts_code', '') for v in holdings.values())
    bt_hb_codes = set(v.get('ts_code', '') for v in bt_hb.values())

    # sells
    rp_sells = set(sells)
    bt_sells = set(bt_sell_reasons.keys())

    # buys (新 pending_orders)
    rp_buys = set(b[2] for b in buys)
    bt_buys_set = set(b['ts_code'] for b in bt_buys)

    dstr = tdate.strftime('%Y-%m-%d')
    hb_ok = rp_hb_codes == bt_hb_codes
    s_ok = rp_sells == bt_sells
    b_ok = rp_buys == bt_buys_set
    all_ok = hb_ok and s_ok and b_ok
    print(f'{dstr:10s} | HB: {"OK" if hb_ok else "DIFF":5s} | '
          f'S: {"OK" if s_ok else "DIFF":5s} | '
          f'B: {"OK" if b_ok else "DIFF":5s} | {"OK" if all_ok else "DIFF!"}')

    if not all_ok:
        print(f'  RP HB: {len(rp_hb_codes)} codes, BT HB: {len(bt_hb_codes)} codes')
        if not hb_ok:
            print(f'  HB diff: RP-BT={sorted(rp_hb_codes - bt_hb_codes)} BT-RP={sorted(bt_hb_codes - rp_hb_codes)}')
        if not s_ok:
            print(f'  RP sells: {sorted(rp_sells)}')
            print(f'  BT sells: {sorted(bt_sells)}')
            print(f'  RP sell_reasons: {dict(sell_reasons)}')
            print(f'  BT sell_reasons: {bt_sell_reasons}')
            # check days_held for sold stocks
            for code in sorted(bt_sells - rp_sells):
                h = holdings.get(code, {})
                bt_hold = bt_hb.get(code, {})
                print(f'  BT sells {code}: RP_dh={h.get("days_held")} BT_dh={bt_hold.get("days_held")} '
                      f'RP_bp={h.get("buy_price")} BT_bp={bt_hold.get("buy_price")}')
            for code in sorted(rp_sells - bt_sells):
                h = holdings.get(code, {})
                bt_hold = bt_hb.get(code, {})
                print(f'  RP sells {code}: RP_dh={h.get("days_held")} BT_dh={bt_hold.get("days_held")} '
                      f'RP_bp={h.get("buy_price")} BT_bp={bt_hold.get("buy_price")}')
        if not b_ok:
            print(f'  RP buys: {sorted(rp_buys)}')
            print(f'  BT buys: {sorted(bt_buys_set)}')
            print(f'  n_avail RP: max_stocks - len(holdings_after_sell)')
        break

    holdings = new_holdings
