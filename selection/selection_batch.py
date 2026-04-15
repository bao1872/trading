#!/usr/bin/env python3
"""
批量选股脚本

Purpose: 批量计算指定日期范围的选股数据
Inputs: 无（自动从数据库获取交易日）
Outputs: stock_selection_results 表
How to Run:
    python selection/selection_batch.py              # 前台运行
    nohup python selection/selection_batch.py > batch.log 2>&1 &  # 后台运行
Side Effects: 写入 stock_selection_results 表
"""
import sys
import subprocess
import os
from datetime import datetime
from sqlalchemy import create_engine, text

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = 'postgresql://bz:es123456@127.0.0.1:5432/bz_stock'
engine = create_engine(DATABASE_URL)

START_DATE = '2025-11-03'
END_DATE = '2026-04-14'


def log(msg):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def main():
    """主函数"""
    log("=" * 60)
    log(f"开始批量选股计算")
    log(f"日期范围: {START_DATE} ~ {END_DATE}")
    log("=" * 60)

    # 获取交易日列表
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT DATE(bar_time) as trade_date
            FROM stock_k_data
            WHERE freq = 'd' AND DATE(bar_time) BETWEEN :start_date AND :end_date
            ORDER BY trade_date
        """), {'start_date': START_DATE, 'end_date': END_DATE})
        dates = [str(row[0]) for row in result]

    log(f"共 {len(dates)} 个交易日需要计算")

    success_count = 0
    fail_count = 0

    for i, d in enumerate(dates, 1):
        log(f"[{i}/{len(dates)}] 计算 {d}...")

        # 删除旧数据
        try:
            with engine.connect() as conn:
                conn.execute(text(f"DELETE FROM stock_selection_results WHERE selection_date = '{d}'"))
                conn.commit()
        except Exception as e:
            log(f"  警告: 删除旧数据失败: {e}")

        # 执行选股计算
        result = subprocess.run(
            ['/usr/bin/python', 'selection/selection_ana.py', d],
            capture_output=True,
            text=True,
            cwd='/root/trading'
        )

        if result.returncode == 0:
            success_count += 1
            # 提取选中数量
            for line in result.stdout.split('\n'):
                if '选中股票数' in line:
                    log(f"  ✓ {line.strip()}")
                    break
        else:
            fail_count += 1
            log(f"  ✗ 失败: {result.stderr[:200]}")

    log("=" * 60)
    log(f"批量计算完成")
    log(f"成功: {success_count} 天, 失败: {fail_count} 天")
    log("=" * 60)


if __name__ == '__main__':
    main()