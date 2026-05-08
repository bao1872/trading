"""
factor_engine.py
因子计算引擎 — 编排 5 大指标组的核心计算

Purpose:
    对单只股票的 K 线数据，编排调用 DSA/BBMACD/Volume Z-score/PAVP/Stop Cluster
    五大指标组的核心计算函数，输出统一格式的因子 DataFrame。
    所有核心计算逻辑来自 features/，本模块只做编排和列名适配。

Inputs:
    - df: pd.DataFrame, 含 open/high/low/close/volume 的日线数据
    - 各指标组参数（可选，有默认值）

Outputs:
    - pd.DataFrame, 原始 K 线列 + 所有因子列

How to Run:
    python market_structure_analysis/factor_engine.py --ts_code 600519.SH --start 2023-01-01

Examples:
    python market_structure_analysis/factor_engine.py --ts_code 600519.SH
    python market_structure_analysis/factor_engine.py --ts_code 000001.SZ --start 2024-01-01 --end 2024-06-01

Side Effects:
    无（只读，不写入数据库或文件）
"""

import argparse
import logging
import sys
import os
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig
from features.volume_zscore_plotly import volume_zscore
from features.pavp_tv_fixed_params_factors import compute_pavp
from features.stop_loss_clustering_with_factors import StopLossClusteringEngine

# 导入 factor_lib 统一入口（自动注册所有因子）
from factor_lib import compute_panel

logger = logging.getLogger(__name__)


def compute_all_factors(
    df: pd.DataFrame,
    dsa_cfg: Optional[DSAConfig] = None,
    bbmacd_rapida: int = 8,
    bbmacd_lenta: int = 26,
    bbmacd_stdv: float = 0.8,
    bbmacd_signal_len: int = 9,
    bbmacd_width_z_window: int = 60,
    vol_z_window: int = 14,
    stop_cluster_model: str = "absorbtion_extremes",
    skip_pavp: bool = False,
) -> pd.DataFrame:
    """
    编排计算所有因子，返回合并后的 DataFrame。

    Parameters
    ----------
    df : pd.DataFrame
        含 open/high/low/close/volume 的日线数据，index 为日期
    dsa_cfg : DSAConfig, optional
        DSA 参数配置，默认 DSAConfig()
    bbmacd_rapida : int
        BBMACD 快线 EMA 周期
    bbmacd_lenta : int
        BBMACD 慢线 EMA 周期
    bbmacd_stdv : float
        BBMACD 布林带标准差倍数
    bbmacd_signal_len : int
        BBMACD 信号线 EMA 周期
    bbmacd_width_z_window : int
        BBMACD 带宽 z-score 回看窗口
    vol_z_window : int
        Volume Z-score 滚动窗口
    stop_cluster_model : str
        Stop Cluster 模型: 'absorbtion_extremes' 或 'volatility_at_entry'
    skip_pavp : bool
        跳过 PAVP 计算（占 77% 单股耗时，仅贡献 2 事件+1 状态）

    Returns
    -------
    pd.DataFrame
        原始 K 线列 + DSA 因子 + BBMACD 因子 + Volume Z-score + PAVP 因子 + Stop Cluster 因子
    """
    if df.empty:
        return df.copy()

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入 DataFrame 缺少必要列: {missing}")

    if dsa_cfg is None:
        dsa_cfg = DSAConfig()

    result = df.copy()

    # 1. 通过 factor_lib 计算所有已注册因子（趋势/位置/动量/量能/结构/节奏/协同/风险）
    logger.debug("通过 factor_lib 计算核心因子...")
    result = compute_panel(result)

    # 2. 补充 vol_zscore 列（event_detector 兼容用）
    logger.debug("计算 Volume Z-score...")
    vol_z, vol_mu, vol_sd = volume_zscore(result["volume"], vol_z_window)
    result["vol_zscore"] = vol_z
    result["vol_zscore_mu"] = vol_mu
    result["vol_zscore_sd"] = vol_sd

    # 3. PAVP（不在 factor_lib 中，保留原有调用）
    if not skip_pavp:
        logger.debug("计算 PAVP 因子...")
        pavp_out, _, _ = compute_pavp(result)
        pavp_factor_cols = [
            col for col in pavp_out.columns
            if col not in result.columns
        ]
        if pavp_factor_cols:
            result = pd.concat([result, pavp_out[pavp_factor_cols]], axis=1)

    # 4. Stop Cluster（不在 factor_lib 中，保留原有调用）
    logger.debug("计算 Stop Cluster 因子...")
    df_for_stop = result.rename(columns={"volume": "vol"})
    stop_args = argparse.Namespace(
        freq="d",
        model=stop_cluster_model,
        show_historical_triggers=False,
        max_lines=20,
    )
    engine = StopLossClusteringEngine(df_for_stop, stop_args)
    engine.run()
    stop_df = engine.df
    stop_factor_cols = [
        col for col in stop_df.columns
        if col not in result.columns and col != "vol"
    ]
    if stop_factor_cols:
        result = pd.concat([result, stop_df[stop_factor_cols]], axis=1)

    return result


def main():
    parser = argparse.ArgumentParser(description="因子计算引擎 — 单股票测试")
    parser.add_argument("--ts_code", type=str, required=True, help="股票代码，如 600519.SH")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--freq", type=str, default="d", help="K线周期")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from datasource.k_data_loader import load_k_data

    logger.info("加载 %s K线数据...", args.ts_code)
    df = load_k_data(args.ts_code, freq=args.freq, start_date=args.start, end_date=args.end)

    if df.empty:
        logger.error("未找到 %s 的K线数据", args.ts_code)
        return

    logger.info("数据量: %d bars, 日期范围: %s ~ %s", len(df), df.index[0], df.index[-1])

    result = compute_all_factors(df)

    factor_cols = [c for c in result.columns if c not in {"open", "high", "low", "close", "volume"}]
    logger.info("计算完成，共 %d 个因子列", len(factor_cols))
    logger.info("因子列: %s", factor_cols)

    print(result.tail(5).to_string())


if __name__ == "__main__":
    main()
