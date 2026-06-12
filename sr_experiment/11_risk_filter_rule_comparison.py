# -*- coding: utf-8 -*-
"""
Purpose: 风险过滤规则对比——手工规则 vs +reclaim_strength过滤 vs +risk_score过滤
Inputs:  分片 parquet + B_cluster_risk 模型
Outputs: 控制台对比表
How to Run:
    python sr_experiment/11_risk_filter_rule_comparison.py --freq w --pivot-len 10
Examples:
    python sr_experiment/11_risk_filter_rule_comparison.py --freq w --pivot-len 10
Side Effects: 无（纯计算+输出）
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sr_experiment.gbdt_config import (
    BAD_BREAK_HORIZON,
    MODELS_DIR,
    PATH_HORIZON,
    SL_PCT_OPP,
    TP_PCT_OPP,
    TRAIN_END,
    VAL_END,
)
from sr_experiment.gbdt_feature_columns import ALL_FEATURE_COLS

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards


def _compute_path_labels_group(group: pd.DataFrame) -> pd.DataFrame:
    close = group["close"].values.astype(float)
    low = group["low"].values.astype(float)
    active_support = group["active_support_ref"].values.astype(float) if "active_support_ref" in group.columns else np.full(len(group), np.nan)
    n = len(group)

    label_tp8 = np.zeros(n, dtype=np.int8)
    label_bad_break = np.zeros(n, dtype=np.int8)
    tp_bar = np.full(n, np.nan, dtype=np.float32)
    sl_bar = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        event_close = close[i]
        if np.isnan(event_close) or event_close <= 0:
            continue

        tp_price = event_close * (1 + TP_PCT_OPP)
        sl_price = event_close * (1 - SL_PCT_OPP)
        support_level = active_support[i] if not np.isnan(active_support[i]) else low[i]

        tp_hit_at = -1
        sl_hit_at = -1
        bad_break_at = -1

        max_k = min(PATH_HORIZON, n - i - 1)
        for k in range(1, max_k + 1):
            fwd_close = close[i + k]
            if np.isnan(fwd_close):
                break
            if tp_hit_at < 0 and fwd_close >= tp_price:
                tp_hit_at = k
            if sl_hit_at < 0 and fwd_close <= sl_price:
                sl_hit_at = k
            if bad_break_at < 0 and k <= BAD_BREAK_HORIZON and fwd_close < support_level:
                bad_break_at = k

        if tp_hit_at > 0 and (sl_hit_at < 0 or tp_hit_at < sl_hit_at):
            label_tp8[i] = 1
        if bad_break_at > 0:
            label_bad_break[i] = 1
        if tp_hit_at > 0:
            tp_bar[i] = tp_hit_at
        if sl_hit_at > 0:
            sl_bar[i] = sl_hit_at

    group = group.copy()
    group["label_tp8_sl6_20"] = label_tp8
    group["label_bad_break_10"] = label_bad_break
    group["tp_hit_bar"] = tp_bar
    group["sl_hit_bar"] = sl_bar
    return group


def _compute_path_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_code" not in df.columns:
        return df
    df = df.sort_values(["ts_code", "bar_time"])
    parts = []
    for _, group in df.groupby("ts_code"):
        parts.append(_compute_path_labels_group(group))
    return pd.concat(parts, ignore_index=True)


def _eval_rule(sub: pd.DataFrame, rule_name: str) -> dict:
    if sub.empty:
        return {"rule": rule_name, "count": 0}
    row = {"rule": rule_name, "count": len(sub)}
    for col in ["fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20"]:
        if col in sub.columns:
            vals = sub[col].dropna()
            row[col] = vals.mean() if len(vals) > 0 else np.nan
    if "fwd_ret_20" in sub.columns:
        vals = sub["fwd_ret_20"].dropna()
        row["win_rate"] = (vals > 0).mean() if len(vals) > 0 else np.nan
    if "fwd_max_ret_20" in sub.columns and "fwd_mdd_20" in sub.columns:
        max_r = sub["fwd_max_ret_20"].dropna()
        mdd = sub["fwd_mdd_20"].dropna().abs()
        if len(max_r) > 0 and len(mdd) > 0 and mdd.mean() > 0:
            row["reward_risk"] = max_r.mean() / mdd.mean()
    if "label_tp8_sl6_20" in sub.columns:
        row["tp_rate"] = sub["label_tp8_sl6_20"].mean()
    if "label_bad_break_10" in sub.columns:
        row["sl_rate"] = sub["label_bad_break_10"].mean()
    if "tp_hit_bar" in sub.columns:
        tp_mask = sub["tp_hit_bar"].notna()
        row["avg_tp_bars"] = sub.loc[tp_mask, "tp_hit_bar"].mean() if tp_mask.any() else np.nan
    if "sl_hit_bar" in sub.columns:
        sl_mask = sub["sl_hit_bar"].notna()
        row["avg_sl_bars"] = sub.loc[sl_mask, "sl_hit_bar"].mean() if sl_mask.any() else np.nan
    return row


def main():
    parser = argparse.ArgumentParser(description="风险过滤规则对比")
    parser.add_argument("--freq", type=str, default="w")
    parser.add_argument("--pivot-len", type=int, default=10)
    args = parser.parse_args()

    cols_needed = list(dict.fromkeys(
        ["ts_code", "bar_time", "close", "low", "high", "active_support_ref",
         "evt_pierce_support_cluster_reclaim_low_volume", "support_reclaim_strength_atr"]
        + ALL_FEATURE_COLS
        + ["fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20", "fwd_reward_risk_20"]
    ))

    print("加载分片数据...")
    parts = []
    for shard_df in iter_shards(args.freq, args.pivot_len, columns=cols_needed, shard_type="panel"):
        parts.append(shard_df)
    if not parts:
        print("无数据")
        return

    all_data = pd.concat(parts, ignore_index=True)
    print(f"总行数: {len(all_data)}")

    if "bar_time" in all_data.columns:
        all_data["bar_time"] = pd.to_datetime(all_data["bar_time"])

    print("计算路径标签...")
    all_data = _compute_path_labels(all_data)

    base_mask = all_data["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)
    df = all_data.loc[base_mask].copy()
    print(f"强簇缩量事件: {len(df)}")

    if "bar_time" in df.columns:
        df["split"] = "train"
        df.loc[df["bar_time"] > TRAIN_END, "split"] = "val"
        df.loc[df["bar_time"] > VAL_END, "split"] = "test"

    model_path = Path(MODELS_DIR) / "B_cluster_risk.txt"
    if not model_path.exists():
        print(f"风险模型不存在: {model_path}")
        return

    model = lgb.Booster(model_file=str(model_path))
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

    print("预测 risk_score...")
    df["risk_score"] = model.predict(df[feature_cols])

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    reclaim_col = "support_reclaim_strength_atr"
    reclaim_threshold = train_df[reclaim_col].median() if reclaim_col in train_df.columns else 0
    print(f"reclaim_strength_atr 阈值(train median): {reclaim_threshold:.4f}")

    risk_threshold = train_df["risk_score"].quantile(0.70)
    print(f"risk_score 阈值(train 70pct): {risk_threshold:.4f}")

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, split_df in splits.items():
        print(f"\n{'='*70}")
        print(f"数据集: {split_name} (N={len(split_df)})")
        print(f"{'='*70}")

        r0 = _eval_rule(split_df, "规则0: 强簇+缩量")
        r1_mask = split_df[reclaim_col] > reclaim_threshold if reclaim_col in split_df.columns else pd.Series(True, index=split_df.index)
        r1 = _eval_rule(split_df[r1_mask], "规则1: +reclaim_strength")
        r2_mask = split_df["risk_score"] < risk_threshold if "risk_score" in split_df.columns else pd.Series(True, index=split_df.index)
        r2 = _eval_rule(split_df[r2_mask], "规则2: +risk_score")

        rows = [r0, r1, r2]
        result_df = pd.DataFrame(rows)

        display_cols = [c for c in ["rule", "count", "fwd_ret_20", "fwd_max_ret_20",
                                     "fwd_mdd_20", "reward_risk", "win_rate",
                                     "tp_rate", "sl_rate", "avg_tp_bars", "avg_sl_bars"]
                        if c in result_df.columns]
        print(tabulate(result_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

        if r0["count"] > 0 and r2["count"] > 0:
            retained = r2["count"] / r0["count"] * 100
            print(f"\n规则2 样本保留率: {retained:.1f}%")

    print("\n\n========== 总结 ==========")
    test_r0 = _eval_rule(test_df, "规则0")
    test_r1_mask = test_df[reclaim_col] > reclaim_threshold if reclaim_col in test_df.columns else pd.Series(True, index=test_df.index)
    test_r1 = _eval_rule(test_df[test_r1_mask], "规则1")
    test_r2_mask = test_df["risk_score"] < risk_threshold if "risk_score" in test_df.columns else pd.Series(True, index=test_df.index)
    test_r2 = _eval_rule(test_df[test_r2_mask], "规则2")

    for rule_name, r in [("规则0", test_r0), ("规则1", test_r1), ("规则2", test_r2)]:
        ret = r.get("fwd_ret_20", np.nan)
        mdd = r.get("fwd_mdd_20", np.nan)
        wr = r.get("win_rate", np.nan)
        tp = r.get("tp_rate", np.nan)
        sl = r.get("sl_rate", np.nan)
        n = r.get("count", 0)
        print(f"{rule_name}: N={n}, ret_20={ret:.4f}, mdd_20={mdd:.4f}, 胜率={wr:.4f}, 先止盈={tp:.4f}, 先止损={sl:.4f}")

    if "bar_time" in df.columns:
        print("\n\n========== 按年份统计 W0/W1/W2 ==========")
        df["year"] = df["bar_time"].dt.year
        year_rows = []
        for year in sorted(df["year"].dropna().unique()):
            ydf = df[df["year"] == year]
            r0 = _eval_rule(ydf, "W0")
            r1_mask = ydf[reclaim_col] > reclaim_threshold if reclaim_col in ydf.columns else pd.Series(True, index=ydf.index)
            r1 = _eval_rule(ydf[r1_mask], "W1")
            r2_mask = ydf["risk_score"] < risk_threshold if "risk_score" in ydf.columns else pd.Series(True, index=ydf.index)
            r2 = _eval_rule(ydf[r2_mask], "W2")
            for rule_label, r in [("W0", r0), ("W1", r1), ("W2", r2)]:
                year_rows.append({
                    "year": int(year),
                    "rule": rule_label,
                    "count": r.get("count", 0),
                    "fwd_ret_20": r.get("fwd_ret_20", np.nan),
                    "win_rate": r.get("win_rate", np.nan),
                    "sl_rate": r.get("sl_rate", np.nan),
                    "reward_risk": r.get("reward_risk", np.nan),
                })
        year_df = pd.DataFrame(year_rows)
        display_cols = [c for c in ["year", "rule", "count", "fwd_ret_20", "win_rate", "sl_rate", "reward_risk"]
                        if c in year_df.columns]
        print(tabulate(year_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
