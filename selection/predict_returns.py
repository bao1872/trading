#!/usr/bin/env python3
"""
选股结果收益率预测脚本

Purpose: 基于Kronos时序模型预测选股结果中每只股票未来3天的收益率
Inputs: stock_selection_results (选股结果表), stock_k_data (K线数据)
Outputs: 预测结果DataFrame (包含未来3天收益率预测和交易建议)
How to Run:
    python selection/predict_returns.py              # 预测当天选股结果
    python selection/predict_returns.py 2026-04-10  # 指定日期预测
Examples:
    python selection/predict_returns.py
    python selection/predict_returns.py 2026-04-20
Side Effects: 无数据库写入，仅输出预测结果到控制台和CSV文件
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

# 设置Kronos路径
KRONOS_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'skills', 'alphaear-predictor', 'scripts')
KRONOS_MODEL_DIR = os.path.join(KRONOS_SCRIPTS_DIR, 'predictor')
if KRONOS_MODEL_DIR not in sys.path:
    sys.path.insert(0, KRONOS_MODEL_DIR)
if KRONOS_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_SCRIPTS_DIR)

# 本地模型路径
LOCAL_TOKENIZER_PATH = os.path.join(KRONOS_MODEL_DIR, 'kronos-tokenizer')
LOCAL_MODEL_PATH = os.path.join(KRONOS_MODEL_DIR, 'kronos-model')

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

# 预测参数
LOOKBACK_DAYS = 60  # 历史数据回看天数
PREDICT_DAYS = 3    # 预测未来天数


def get_selection_results(selection_date: date) -> pd.DataFrame:
    """获取指定日期的选股结果"""
    sql = """
    SELECT 
        selection_date,
        ts_code,
        stock_name,
        weekly_reversal_buy,
        weekly_breakout_buy,
        weekly_bb_width_zscore,
        weekly_vol_zscore,
        weekly_vwap_deviation
    FROM stock_selection_results 
    WHERE selection_date = :selection_date
    ORDER BY ts_code
    """
    df = pd.read_sql(text(sql), engine, params={'selection_date': selection_date})
    return df


def get_kline_data(ts_code: str, end_date: date, bars: int = 120) -> pd.DataFrame:
    """获取股票K线数据"""
    symbol = str(ts_code).strip().upper().split('.')[0]
    
    sql = """
    SELECT 
        bar_time,
        open,
        high,
        low,
        close,
        volume
    FROM stock_k_data
    WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) 
      AND freq = 'd'
      AND DATE(bar_time) <= :end_date
    ORDER BY bar_time DESC
    LIMIT :bars
    """
    params = {
        'ts_code': symbol,
        'ts_code_sh': f'{symbol}.SH',
        'ts_code_sz': f'{symbol}.SZ',
        'end_date': end_date.strftime('%Y-%m-%d'),
        'bars': bars
    }
    
    df = pd.read_sql(text(sql), engine, params=params)
    if not df.empty:
        df = df.sort_values('bar_time').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['bar_time'])
    return df


class LocalKronosPredictor:
    """使用本地Kronos模型的预测器"""
    
    def __init__(self, device: Optional[str] = None):
        import torch
        from loguru import logger
        
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self._predictor = None
        
        try:
            logger.info(f"🔮 Loading Kronos Model from local path on {device}...")
            
            # 导入模型类
            from model.kronos import Kronos, KronosTokenizer
            from model import KronosPredictor
            from schema.models import KLinePoint
            
            # 检查本地模型路径
            if not os.path.exists(LOCAL_TOKENIZER_PATH):
                raise FileNotFoundError(f"Tokenizer path not found: {LOCAL_TOKENIZER_PATH}")
            if not os.path.exists(LOCAL_MODEL_PATH):
                raise FileNotFoundError(f"Model path not found: {LOCAL_MODEL_PATH}")
            
            logger.info(f"  Tokenizer: {LOCAL_TOKENIZER_PATH}")
            logger.info(f"  Model: {LOCAL_MODEL_PATH}")
            
            # 加载本地模型
            tokenizer = KronosTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH, local_files_only=True)
            model = Kronos.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
            
            tokenizer = tokenizer.to(device)
            model = model.to(device)
            
            self._predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
            self.KLinePoint = KLinePoint
            logger.info("✅ Kronos Model loaded successfully from local path.")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Kronos Model: {e}")
            import traceback
            traceback.print_exc()
            self._predictor = None
    
    def get_base_forecast(self, df: pd.DataFrame, lookback: int = 20, pred_len: int = 5, news_text: Optional[str] = None) -> List:
        """生成原始模型预测"""
        from pandas.tseries.offsets import BusinessDay
        from loguru import logger
        
        if self._predictor is None:
            logger.error("Predictor not initialized.")
            return []
        
        if len(df) < lookback:
            logger.warning(f"Insufficient historical data ({len(df)}) for lookback ({lookback}).")
            return []
        
        # 获取最后 lookback 条数据
        x_df = df.iloc[-lookback:].copy()
        x_timestamp = pd.to_datetime(x_df['date'])
        last_date = x_timestamp.iloc[-1]
        
        # 生成未来时间戳
        future_dates = pd.date_range(start=last_date + BusinessDay(1), periods=pred_len, freq='B')
        y_timestamp = pd.Series(future_dates)
        
        try:
            # 预测所需的列
            cols = ['open', 'high', 'low', 'close', 'volume']
            pred_df = self._predictor.predict(
                df=x_df[cols],
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0, 
                top_p=0.9, 
                sample_count=1,
                verbose=False,
                news_emb=None
            )
            
            # 转换为 KLinePoint
            results = []
            for date, row in pred_df.iterrows():
                results.append(self.KLinePoint(
                    date=date.strftime("%Y-%m-%d"),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                ))
            return results
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []


def predict_stock_returns(
    predictor: LocalKronosPredictor,
    ts_code: str,
    stock_name: str,
    end_date: date,
    lookback: int = LOOKBACK_DAYS
) -> Optional[Dict]:
    """
    预测单只股票未来3天收益率
    
    Returns:
        {
            'ts_code': 股票代码,
            'stock_name': 股票名称,
            'current_price': 当前收盘价,
            'pred_day1_return': 第1天预测收益率,
            'pred_day2_return': 第2天预测收益率,
            'pred_day3_return': 第3天预测收益率,
            'pred_cum_return_3d': 3天累计预测收益率,
            'confidence': 预测置信度 (0-1),
        }
    """
    # 获取历史K线数据
    df = get_kline_data(ts_code, end_date, bars=lookback + 10)
    
    if len(df) < lookback * 0.8:  # 至少需要80%的数据
        return None
    
    # 取最近lookback天数据
    df = df.tail(lookback).copy()
    
    if len(df) < 20:  # 最少需要20天数据
        return None
    
    try:
        # 使用Kronos预测未来3天
        forecast = predictor.get_base_forecast(
            df=df,
            lookback=min(lookback, len(df)),
            pred_len=PREDICT_DAYS,
            news_text=None  # 不使用新闻情感
        )
        
        if not forecast or len(forecast) < PREDICT_DAYS:
            return None
        
        # 获取当前收盘价
        current_price = df['close'].iloc[-1]
        
        # 计算每日预测收益率
        pred_prices = [p.close for p in forecast[:PREDICT_DAYS]]
        
        day1_return = (pred_prices[0] - current_price) / current_price if current_price > 0 else 0
        day2_return = (pred_prices[1] - pred_prices[0]) / pred_prices[0] if pred_prices[0] > 0 else 0
        day3_return = (pred_prices[2] - pred_prices[1]) / pred_prices[1] if pred_prices[1] > 0 else 0
        
        # 3天累计收益率
        cum_return = (pred_prices[2] - current_price) / current_price if current_price > 0 else 0
        
        # 计算预测置信度 (基于历史波动率)
        hist_returns = df['close'].pct_change().dropna()
        volatility = hist_returns.std()
        confidence = max(0, min(1, 1 - volatility * 10))  # 波动率越低，置信度越高
        
        return {
            'ts_code': ts_code,
            'stock_name': stock_name,
            'current_price': round(current_price, 2),
            'pred_day1_return': round(day1_return * 100, 2),  # 转为百分比
            'pred_day2_return': round(day2_return * 100, 2),
            'pred_day3_return': round(day3_return * 100, 2),
            'pred_cum_return_3d': round(cum_return * 100, 2),
            'confidence': round(confidence, 2),
            'pred_price_day1': round(pred_prices[0], 2),
            'pred_price_day2': round(pred_prices[1], 2),
            'pred_price_day3': round(pred_prices[2], 2),
        }
        
    except Exception as e:
        print(f"  预测失败 {ts_code}: {e}")
        return None


def generate_trading_advice(
    cum_return: float,
    confidence: float,
    weekly_reversal: bool,
    weekly_breakout: bool,
    bb_width_zscore: Optional[float] = None
) -> str:
    """
    生成交易建议
    
    规则:
    - 强烈买入: 3天累计收益 > 5% 且 (周线反转买点或周线突破买点) 且 置信度 >= 0.5
    - 买入: 3天累计收益 > 3% 且 置信度 >= 0.4
    - 观望: 3天累计收益在 -3% ~ 3% 之间
    - 回避: 3天累计收益 < -3%
    """
    has_bsm_signal = weekly_reversal or weekly_breakout
    
    if cum_return > 5 and has_bsm_signal and confidence >= 0.5:
        return "强烈买入"
    elif cum_return > 3 and confidence >= 0.4:
        return "买入"
    elif cum_return < -3:
        return "回避"
    else:
        return "观望"


def predict_selection_returns(selection_date: date, max_stocks: Optional[int] = None) -> pd.DataFrame:
    """
    预测选股结果中所有股票的未来收益率
    
    Args:
        selection_date: 选股日期
        max_stocks: 最多预测股票数量（用于测试）
    """
    print("=" * 80)
    print("选股结果收益率预测 (Kronos模型)")
    print("=" * 80)
    print(f"选股日期: {selection_date}")
    print(f"预测未来: {PREDICT_DAYS} 天")
    print("-" * 80)
    
    # 1. 获取选股结果
    selection_df = get_selection_results(selection_date)
    if selection_df.empty:
        print(f"未找到 {selection_date} 的选股结果")
        return pd.DataFrame()
    
    print(f"选股结果数量: {len(selection_df)}")
    
    if max_stocks:
        selection_df = selection_df.head(max_stocks)
        print(f"本次预测数量: {len(selection_df)} (限制模式)")
    
    # 2. 初始化Kronos预测器
    print("\n正在加载Kronos模型...")
    try:
        predictor = LocalKronosPredictor()
        if predictor._predictor is None:
            print("Kronos模型加载失败，请检查模型配置")
            return pd.DataFrame()
        print("Kronos模型加载成功")
    except Exception as e:
        print(f"Kronos模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # 3. 逐只预测
    print(f"\n开始预测 {len(selection_df)} 只股票...")
    predictions = []
    
    for idx, row in tqdm(selection_df.iterrows(), total=len(selection_df), desc="预测进度"):
        ts_code = row['ts_code']
        stock_name = row['stock_name'] or ''
        
        pred = predict_stock_returns(
            predictor=predictor,
            ts_code=ts_code,
            stock_name=stock_name,
            end_date=selection_date
        )
        
        if pred:
            # 添加BSM信号信息
            pred['weekly_reversal_buy'] = row['weekly_reversal_buy']
            pred['weekly_breakout_buy'] = row['weekly_breakout_buy']
            pred['weekly_bb_width_zscore'] = row['weekly_bb_width_zscore']
            pred['weekly_vwap_deviation'] = row['weekly_vwap_deviation']
            
            # 生成交易建议
            pred['trading_advice'] = generate_trading_advice(
                cum_return=pred['pred_cum_return_3d'],
                confidence=pred['confidence'],
                weekly_reversal=row['weekly_reversal_buy'],
                weekly_breakout=row['weekly_breakout_buy'],
                bb_width_zscore=row['weekly_bb_width_zscore']
            )
            
            predictions.append(pred)
    
    if not predictions:
        print("没有成功的预测结果")
        return pd.DataFrame()
    
    # 4. 整理结果
    result_df = pd.DataFrame(predictions)
    
    # 按3天累计收益率排序
    result_df = result_df.sort_values('pred_cum_return_3d', ascending=False).reset_index(drop=True)
    
    print(f"\n预测完成: {len(result_df)} / {len(selection_df)} 只股票成功")
    
    return result_df


def display_results(df: pd.DataFrame, top_n: int = 20):
    """显示预测结果"""
    if df.empty:
        return
    
    print("\n" + "=" * 80)
    print("预测结果汇总")
    print("=" * 80)
    
    # 交易建议统计
    advice_counts = df['trading_advice'].value_counts()
    print("\n交易建议分布:")
    for advice, count in advice_counts.items():
        print(f"  {advice}: {count} 只")
    
    # 收益率统计
    print("\n收益率统计:")
    print(f"  3天累计收益最高: {df['pred_cum_return_3d'].max():.2f}%")
    print(f"  3天累计收益最低: {df['pred_cum_return_3d'].min():.2f}%")
    print(f"  3天累计收益平均: {df['pred_cum_return_3d'].mean():.2f}%")
    print(f"  3天累计收益中位数: {df['pred_cum_return_3d'].median():.2f}%")
    
    # 显示Top N
    print(f"\n前 {top_n} 名股票 (按3天累计收益率排序):")
    print("-" * 80)
    display_cols = [
        'ts_code', 'stock_name', 'current_price',
        'pred_day1_return', 'pred_day2_return', 'pred_day3_return',
        'pred_cum_return_3d', 'confidence', 'trading_advice'
    ]
    print(df[display_cols].head(top_n).to_string(index=False))
    
    # 显示强烈买入的股票
    strong_buy = df[df['trading_advice'] == '强烈买入']
    if not strong_buy.empty:
        print(f"\n强烈买入推荐 ({len(strong_buy)} 只):")
        print("-" * 80)
        print(strong_buy[display_cols].to_string(index=False))


def save_results(df: pd.DataFrame, selection_date: date):
    """保存预测结果到CSV"""
    if df.empty:
        return
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"predict_returns_{selection_date.strftime('%Y%m%d')}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存: {filepath}")


def parse_date(date_str: str) -> date:
    """解析日期字符串"""
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='选股结果收益率预测工具（基于Kronos时序模型）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/predict_returns.py                    # 预测当天选股结果
  python selection/predict_returns.py 2026-04-20         # 指定日期预测
  python selection/predict_returns.py --max-stocks 10    # 只预测前10只股票（测试用）
        """
    )
    parser.add_argument(
        'date',
        nargs='?',
        help='选股日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认为当天'
    )
    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='最多预测股票数量（用于测试）'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='显示前N名股票，默认20'
    )
    
    args = parser.parse_args()
    
    # 解析日期
    if args.date:
        try:
            selection_date = parse_date(args.date)
        except ValueError as e:
            print(f"错误: {e}")
            print("日期格式应为: YYYY-MM-DD 或 YYYYMMDD")
            sys.exit(1)
    else:
        selection_date = date.today()
    
    # 执行预测
    result_df = predict_selection_returns(
        selection_date=selection_date,
        max_stocks=args.max_stocks
    )
    
    # 显示结果
    display_results(result_df, top_n=args.top_n)
    
    # 保存结果
    save_results(result_df, selection_date)
    
    print("\n" + "=" * 80)
    print("预测完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
