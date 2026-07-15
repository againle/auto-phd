"""src package — 核心模块"""
from pathlib import Path


def ensure_directories():
    """确保所有数据目录存在"""
    dirs = [
        "data/professors",
        "data/cache",
        "data/checkpoints",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
