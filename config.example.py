import os
import socket

# Tushare API Token
# 请替换为你自己的 Tushare Pro API Token
TS_TOKEN = "your_tushare_token_here"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def _is_local_postgres_available(host: str = "127.0.0.1", port: int = 5432, timeout: float = 2.0) -> bool:
    """检测本地 PostgreSQL 服务是否可用
    
    Args:
        host: 主机地址，默认 127.0.0.1（避免 DNS 解析延迟）
        port: 端口，默认 5432
        timeout: 连接超时时间（秒）
    
    Returns:
        True if 本地 PostgreSQL 可用，否则 False
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_database_url() -> str:
    """获取数据库连接 URL
    
    优先级：
    1. 环境变量 DATABASE_URL（如果设置）
    2. 本地 PostgreSQL (127.0.0.1:5432) 如果可用
    3. 远程 PostgreSQL (配置中的默认地址)
    
    Returns:
        PostgreSQL 连接 URL
    """
    # 1. 环境变量优先
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    
    # 2. 尝试本地连接
    if _is_local_postgres_available():
        return "postgresql://username:password@127.0.0.1:5432/dbname"
    
    # 3. 默认使用远程（请修改为你的远程数据库地址）
    return "postgresql://username:password@your_server_ip:5432/dbname"


# 动态获取数据库 URL
DATABASE_URL = get_database_url()
