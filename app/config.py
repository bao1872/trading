# -*- coding: utf-8 -*-
"""
项目配置文件
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_URL = 'postgresql://bz:UG8dSLGOdOkk@192.168.31.188:5432/stock'

STOCK_POOL_PATH = os.path.join(PROJECT_ROOT, 'stock_concepts_cache.xlsx')

PYTDX_SERVERS = [
    ("119.147.212.81", 7709),
    ("119.147.164.60", 7709),
    ("14.215.128.18", 7709),
    ("14.215.128.116", 7709),
    ("101.133.156.38", 7709),
    ("114.80.149.19", 7709),
    ("115.238.90.165", 7709),
    ("123.125.108.23", 7709),
    ("180.153.18.170", 7709),
    ("202.108.253.131", 7709),
]

SUPPORTED_FREQUENCIES = ['15m', '60m', 'd', 'w']

AMP_CANDIDATE_PERIODS = [50, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 200, 220, 250, 280, 310, 340, 370, 400]

# 飞书消息配置
FEISHU_APP_ID = 'cli_a6b37d1d077b900e'
FEISHU_APP_SECRET = 'ZVNeWFZuBuftTh3WGtcAIepdAeBbiX5Z'
FEISHU_USER_ID = 'bg332537'  # 待填充：运行 get_feishu_userid.py 获取后填写

if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATABASE_URL: {DATABASE_URL}")
    print(f"STOCK_POOL_PATH: {STOCK_POOL_PATH}")
    print(f"SUPPORTED_FREQUENCIES: {SUPPORTED_FREQUENCIES}")
