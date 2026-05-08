# -*- coding: utf-8 -*-
"""
event_lib/states.py - 事件序列状态判定

Purpose: 基于连续事件推断个股所处阶段。

States:
    - 主升阶段: 连续3次放量上涨事件
    - 调整阶段: 趋势翻转+缩量
    - 下跌阶段: 连续3次放量下跌事件
    - 筑底阶段: 低点+缩量止跌
    - 震荡阶段: 无明显趋势

Usage:
    from event_lib.states import infer_state
    state = infer_state(events_df)
"""
import pandas as pd


def infer_state(events_df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    基于事件序列推断当前所处阶段。

    Args:
        events_df: 包含事件列的 DataFrame
        window: 观察窗口（bars）

    Returns:
        状态序列
    """
    states = []

    for i in range(len(events_df)):
        if i < window:
            states.append("unknown")
            continue

        window_df = events_df.iloc[i - window + 1 : i + 1]

        # 统计窗口内各类事件次数
        up_moves = window_df.get("evt_up_move_with_vol_spike", pd.Series(0)).sum()
        down_moves = window_df.get("evt_down_move_with_vol_spike", pd.Series(0)).sum()
        flips = (
            window_df.get("evt_dsa_dir_flip_up", pd.Series(0)).sum()
            + window_df.get("evt_dsa_dir_flip_down", pd.Series(0)).sum()
        )
        vol_shrinks = window_df.get("evt_vol_shrink", pd.Series(0)).sum()
        low_vol_shrink = window_df.get("evt_low_with_vol_shrink", pd.Series(0)).sum()

        if up_moves >= 3:
            states.append("主升阶段")
        elif down_moves >= 3:
            states.append("下跌阶段")
        elif low_vol_shrink >= 1 or (flips >= 1 and vol_shrinks >= 2):
            states.append("筑底阶段")
        elif flips >= 2:
            states.append("调整阶段")
        else:
            states.append("震荡阶段")

    return pd.Series(states, index=events_df.index)
