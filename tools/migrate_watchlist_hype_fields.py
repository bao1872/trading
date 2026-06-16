"""
迁移脚本：为 stock_watchlist 表新增 hype_logic 字段

Purpose: 为自选股表添加炒作逻辑字段
Inputs/Outputs: 修改 stock_watchlist 表结构
How to Run:
    python tools/migrate_watchlist_hype_fields.py
    python tools/migrate_watchlist_hype_fields.py --dry-run
Examples:
    python tools/migrate_watchlist_hype_fields.py
    python tools/migrate_watchlist_hype_fields.py --dry-run
Side Effects: ALTER TABLE stock_watchlist ADD COLUMN（幂等，字段已存在则跳过）
"""

import argparse
import logging
import sys
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text
from datasource.database import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def migrate(dry_run: bool = False) -> None:
    """为 stock_watchlist 添加 hype_logic 字段（幂等）"""
    alter_sql = text("""
        ALTER TABLE stock_watchlist
        ADD COLUMN IF NOT EXISTS hype_logic VARCHAR(100) DEFAULT ''
    """)

    if dry_run:
        logger.info("[DRY-RUN] 将执行: %s", alter_sql.text)
        return

    with get_session() as session:
        # 检查字段是否已存在
        check_sql = text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'stock_watchlist' AND column_name = 'hype_logic'
        """)
        result = session.execute(check_sql).fetchone()
        if result:
            logger.info("字段 hype_logic 已存在，跳过")
            return

        session.execute(alter_sql)
        session.commit()
        logger.info("已添加 hype_logic 字段到 stock_watchlist 表")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 stock_watchlist 添加 hype_logic 字段")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的SQL，不实际执行")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
