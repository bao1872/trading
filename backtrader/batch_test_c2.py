# -*- coding: utf-8 -*-
"""
批量测试 C2 策略（50只随机股票）
"""
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasource.database import get_engine


def get_random_stocks(n=50, seed=42):
    """从数据库随机抽取n只股票"""
    engine = get_engine()
    query = f"""
    SELECT ts_code as code FROM stock_pools
    ORDER BY RANDOM()
    LIMIT {n}
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df['code'].tolist()


def run_strategy(symbol, bars=1200):
    """运行单只股票策略"""
    cmd = [
        'python', 'c2_main_strategy.py',
        '--symbol', symbol,
        '--bars', str(bars),
        '--out-trades', f'c2_batch_trades/{symbol}_trades.csv',
        '--out-summary', f'c2_batch_trades/{symbol}_summary.csv'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr[:200]
    except Exception as e:
        return False, str(e)


def main():
    # 创建输出目录
    output_dir = Path('c2_batch_trades')
    output_dir.mkdir(exist_ok=True)
    
    # 获取50只随机股票
    print("获取50只随机股票...")
    stocks = get_random_stocks(50, seed=42)
    print(f"选中股票: {stocks[:10]}... (共{len(stocks)}只)")
    
    # 批量运行
    results = []
    success_count = 0
    
    for i, symbol in enumerate(stocks, 1):
        print(f"\n[{i}/50] 测试 {symbol}...", end=' ')
        success, error = run_strategy(symbol)
        
        if success:
            # 读取汇总结果
            summary_file = output_dir / f"{symbol}_summary.csv"
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                if not df.empty and df['trade_n'].iloc[0] > 0:
                    results.append({
                        'symbol': symbol,
                        'trade_n': int(df['trade_n'].iloc[0]),
                        'ret_mean': float(df['ret_mean'].iloc[0]),
                        'wr': float(df['wr'].iloc[0]),
                        'rr_mean': float(df['rr_mean'].iloc[0]),
                    })
                    print(f"✓ {int(df['trade_n'].iloc[0])}笔交易")
                    success_count += 1
                else:
                    print("✓ 无交易信号")
            else:
                print("✓ 无输出")
        else:
            print(f"✗ 失败: {error[:50]}")
    
    # 汇总分析
    print(f"\n{'='*60}")
    print("批量测试结果汇总")
    print(f"{'='*60}")
    print(f"成功运行: {success_count}/50 只股票")
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"有交易的个股: {len(df_results)} 只")
        print(f"总交易次数: {df_results['trade_n'].sum()}")
        print(f"\n收益统计:")
        print(f"  平均收益: {df_results['ret_mean'].mean():.2%}")
        print(f"  收益中位数: {df_results['ret_mean'].median():.2%}")
        print(f"  胜率均值: {df_results['wr'].mean():.1%}")
        print(f"  RR均值: {df_results['rr_mean'].mean():.2f}")
        print(f"  正收益个股: {(df_results['ret_mean'] > 0).sum()}/{len(df_results)}")
        
        # 保存汇总
        df_results.to_csv('c2_batch_summary.csv', index=False)
        print(f"\n详细结果已保存: c2_batch_summary.csv")
        print(f"个股交易明细: c2_batch_trades/")
    else:
        print("无有效交易数据")


if __name__ == "__main__":
    main()
