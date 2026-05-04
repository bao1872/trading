#!/usr/bin/env python3
"""
批量选股脚本 - 后台运行

Purpose: 对指定日期范围进行批量选股并保存到数据库
Usage:   python batch_selection.py 2026-04-01 2026-04-13
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta
from selection.selection_ana import main as selection_main
import subprocess


def batch_selection(start_date_str: str, end_date_str: str):
    """批量选股"""
    start_date = date.fromisoformat(start_date_str)
    end_date = date.fromisoformat(end_date_str)
    
    print(f"开始批量选股: {start_date} 到 {end_date}")
    print("=" * 60)
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n处理日期: {date_str}")
        
        try:
            # 调用选股脚本
            result = subprocess.run(
                [sys.executable, "selection/selection_ana.py", date_str],
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                print(f"✅ {date_str} 选股完成")
            else:
                print(f"❌ {date_str} 选股失败: {result.stderr}")
        except Exception as e:
            print(f"❌ {date_str} 异常: {e}")
        
        current_date += timedelta(days=1)
    
    print("\n" + "=" * 60)
    print("批量选股完成")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_selection.py <start_date> <end_date>")
        print("Example: python batch_selection.py 2026-04-01 2026-04-13")
        sys.exit(1)
    
    batch_selection(sys.argv[1], sys.argv[2])
